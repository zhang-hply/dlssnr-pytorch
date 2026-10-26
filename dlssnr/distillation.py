"""Quantization-aware recovery of a fixed depth/width pruning structure.

The inference modules keep decoded weights as buffers. A student owns a fresh
network, promotes only the retained learned buffers to float32 master parameters,
and parametrizes them onto the inference formats. Export materializes the original
buffer names so an ordinary ``NRNet`` can load the result without training code.
"""
from __future__ import annotations

import copy
import functools
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint

from . import int4, int8, quant
from .ablation import layer_macs, units
from .io import Controls
from .layers import decoder, pre_post, split_swin, swin, vit
from .model import NRNet, _main
from .plan import LayerSpec, build_plan
from .pruning import PruningSpec, validate_spec
from .training import distillation_mode, fp8_ste, fp16_ste
from .width_pruning import apply_spec, validate_applied_spec


def validate_pruning(plan: Sequence[LayerSpec], dropped_blocks: Iterable[int],
                     keep_ratio: float = 0.8) -> dict:
    """Validate whole-block safety and the remaining matrix-MAC budget."""
    if isinstance(keep_ratio, bool) or not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be finite and within (0, 1]")
    blocks = list(dropped_blocks)
    if any(isinstance(block, bool) or not isinstance(block, int) for block in blocks):
        raise ValueError("dropped blocks must be integer block numbers")
    if len(blocks) != len(set(blocks)):
        raise ValueError("dropped blocks must not contain duplicates")
    candidates = units(plan)
    unsafe = sorted(set(blocks) - candidates.keys())
    if unsafe:
        raise ValueError(f"unsafe or unknown whole-block pruning candidates: {unsafe}")
    blocks.sort()
    total = sum(layer_macs(plan).values())
    if total <= 0:
        raise ValueError("the model plan must have a positive MAC count")
    remaining = total - sum(candidates[block].macs for block in blocks)
    if remaining > total * keep_ratio + 1e-6:
        raise ValueError(f"pruned model retains {remaining / total:.6%} MACs, "
                         f"exceeding the {keep_ratio:.6%} budget")
    return {"blocks": blocks,
            "layer_indices": sorted(index for block in blocks for index in candidates[block].indices),
            "total_macs": total, "macs": remaining, "keep_ratio": remaining / total}


def read_pruning_report(path: str | Path, plan: Sequence[LayerSpec],
                        keep_ratio: float = 0.8, *, plan_method: str = "best",
                        weights_hash: str | None = None, controls: dict | None = None) -> dict:
    """Read an ablation report and revalidate the selected structure and MACs.

    MAC counts and layer indices from the report are never trusted. When present,
    its original image dimensions must describe exactly the supplied plan.
    """
    with Path(path).open(encoding="utf-8") as handle:
        report = json.load(handle)
    if plan_method not in {"best", "depth", "width", "hybrid"}:
        raise ValueError("plan_method must be best, depth, width or hybrid")
    if not isinstance(report, dict) or type(report.get("schema_version", 1)) is not int \
            or report.get("schema_version", 1) not in (1, 2):
        raise ValueError("unsupported pruning report schema")
    version = report.get("schema_version", 1)
    metadata = report.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("pruning report metadata must be an object")
    if "width" in metadata or "height" in metadata:
        width, height = metadata.get("width"), metadata.get("height")
        if (isinstance(width, bool) or not isinstance(width, int)
                or isinstance(height, bool) or not isinstance(height, int)):
            raise ValueError("pruning report dimensions must be integer width and height")
        if list(plan) != build_plan(width, height):
            raise ValueError("pruning report dimensions do not match the student's model plan")
    if weights_hash is not None and metadata.get("weights_sha256") not in (None, weights_hash):
        raise ValueError("pruning report was searched with different source weights")
    if controls is not None and metadata.get("controls") not in (None, controls):
        raise ValueError("pruning report controls differ from the teacher cache and training controls")
    if version == 1:
        if plan_method not in {"best", "depth"}:
            raise ValueError(f"legacy pruning report has no {plan_method} method")
        row = report.get("best")
        if not isinstance(row, dict):
            raise ValueError("pruning report has no best combination")
        if "pruning_spec" in row:
            raise ValueError("structured pruning requires a schema-v2 report")
        blocks = row.get("blocks")
        if not isinstance(blocks, list):
            raise ValueError("pruning report best.blocks must be a list")
        return validate_pruning(plan, blocks, keep_ratio)
    if plan_method == "best":
        row = report.get("best")
    else:
        methods = report.get("methods", {})
        method = methods.get(plan_method) if isinstance(methods, dict) else None
        row = method.get("best") if isinstance(method, dict) else None
    if not isinstance(row, dict):
        raise ValueError(f"pruning report has no feasible {plan_method} combination")
    if "pruning_spec" not in row:
        raise ValueError("schema-v2 pruning result is missing pruning_spec")
    selected = PruningSpec.from_dict(row["pruning_spec"])
    if plan_method == "depth" and selected.widths:
        raise ValueError("depth result must not include width pruning")
    if plan_method == "width" and selected.dropped_blocks:
        raise ValueError("width result must not include dropped blocks")
    result = validate_spec(plan, selected, keep_ratio)
    result["plan_method"] = plan_method
    return result


class _InferenceFormat(nn.Module):
    def __init__(self, format_: str):
        super().__init__()
        self.format = format_

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.format == "fp8":
            return fp8_ste(value)
        if self.format == "fp16":
            return fp16_ste(value)
        return value


def _formats(module: nn.Module) -> dict[str, str]:
    """Explicit source formats; fail closed when inference adds learned buffers."""
    if isinstance(module, swin.SwinParams):
        return {**dict.fromkeys(("expand", "mid", "contract", "qkv", "out_proj"), "fp8"),
                **dict.fromkeys(("mlp_scale", "attn_scale"), "fp16"),
                **dict.fromkeys(("exp_bias", "head_scale"), "fp32")}
    if isinstance(module, pre_post.PreBlock):
        return {"lift": "fp16"}
    if isinstance(module, pre_post.PostBlock):
        return dict.fromkeys(("main_gain", "skip_gain", "project"), "fp16")
    if isinstance(module, swin.SwinLayer):
        return {"resample": "fp8", "gain": "fp16"}
    if isinstance(module, split_swin.SplitFfwd):
        return dict.fromkeys(("expand_in", "expand_hidden", "contract"), "fp8")
    if isinstance(module, split_swin._Projection):
        return {"weight": "fp8", "gain": "fp16"}
    if isinstance(module, split_swin.SplitQKVAttn):
        return {"qkv": "fp8", "bias": "fp16", "scale": "fp32"}
    if isinstance(module, vit._Matmul):
        return {"weight": "fp8"}
    if isinstance(module, (vit.VitFfnContract, decoder.DecInputUpsample)):
        return {"gain": "fp16"}
    if isinstance(module, vit.VitQKV):
        return {"temperature": "fp32"}
    return {}


def require_reference(net: NRNet | None = None) -> None:
    """Reject alternate arithmetic when preparing, training or scoring targets."""
    if int4.active() or int8.sites or quant.skip_blocks:
        raise ValueError("distillation requires the reference FP8 model; disable INT4, INT8, "
                         "and FP8-skip experiments")
    # Disabling a global flag after constructing an INT4 network does not remove
    # its recovered matrices or the alternate activation path.
    if net is not None and any(isinstance(module, (int4.Int4Matrix, int4.VitFfn, int4.SwinSite))
                               for module in net.modules()):
        raise ValueError("construct a fresh reference network without INT4 recovery modules")


class DistilledStudent(nn.Module):
    """Own and train a fresh inference network with a fixed safe pruning mask.

    ``forward`` accepts the same padded float32 tensor as ``NRNet.run`` and
    returns its cropped float output. The teacher must be a separate network.
    All activation checkpoint inputs/outputs include the real external residual
    branches; recomputation enters the same training context as the first pass.
    """

    def __init__(self, net: NRNet, dropped_blocks: Iterable[int] = (),
                 checkpoint_blocks: bool = True, controls: Controls | None = None,
                 keep_ratio: float = 0.8, *, pruning_spec: PruningSpec | None = None):
        super().__init__()
        require_reference(net)
        dropped_blocks = tuple(dropped_blocks)
        if pruning_spec is not None and dropped_blocks:
            raise ValueError("supply either dropped_blocks or pruning_spec, not both")
        self.pruning_spec = pruning_spec
        self.pruning = (validate_pruning(net.plan, dropped_blocks, keep_ratio) if pruning_spec is None else
                        validate_spec(net.plan, pruning_spec, keep_ratio))
        expected_skip = set(self.pruning["layer_indices"])
        if net.skip and net.skip != expected_skip:
            raise ValueError("student network has an existing incompatible or partial skip mask")
        if list(net.parameters()):
            raise ValueError("student must own a fresh inference network with buffer weights")
        if pruning_spec is not None:
            apply_spec(net, pruning_spec)
        self.net = net
        self.controls = copy.deepcopy(controls if controls is not None else Controls())
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self._canonical_keys = tuple(net.state_dict())
        net.skip = expected_skip
        self._promote_weights()
        groups: list[list[LayerSpec]] = []
        for spec in net.plan:
            if not groups or groups[-1][0].block != spec.block:
                groups.append([])
            groups[-1].append(spec)
        self._groups = tuple(tuple(group) for group in groups)

    def _promote_weights(self) -> None:
        for spec, layer in zip(self.net.plan, self.net.layers):
            if spec.index in self.net.skip:
                continue
            # Snapshot modules before registration adds parametrization children.
            for name, module in list(layer.named_modules()):
                formats = _formats(module)
                for buffer_name, value in list(module._buffers.items()):
                    if value is None:
                        continue
                    if buffer_name not in formats:
                        raise ValueError(f"unclassified retained buffer: layers.{spec.index}."
                                         f"{name + '.' if name else ''}{buffer_name}")
                    if not value.is_floating_point():
                        raise ValueError(f"learned buffer {buffer_name} must be floating point")
                    master = nn.Parameter(value.detach().to(torch.float32).clone())
                    del module._buffers[buffer_name]
                    module.register_parameter(buffer_name, master)
                    parametrize.register_parametrization(
                        module, buffer_name, _InferenceFormat(formats[buffer_name]))

    def _run_group(self, specs: tuple[LayerSpec, ...], external_indices: tuple[int, ...],
                   retained_indices: tuple[int, ...], previous, image: torch.Tensor, *external):
        outputs = dict(zip(external_indices, external))
        int8.current = quant.current_block = specs[0].block
        for spec in specs:
            layer = self.net.layers[spec.index]
            if hasattr(layer, "controls"):
                layer.controls = self.controls
            source = self.net.skip_source[spec.index]
            skip = None if source is None else outputs[source]
            if spec.kind == "PreBlock":
                value = layer(image)
            elif spec.kind == "PostBlock":
                value = layer(_main(previous), skip[1], image)
            elif spec.kind == "SplitFfwdProj":
                value = layer(previous)
            else:
                if spec.kind == "DecInputUpsample":
                    skip = _main(skip)
                value = layer(_main(previous), skip)
            outputs[spec.index] = previous = value
        return tuple(outputs[index] for index in retained_indices)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        require_reference(self.net)
        shape = (1, 3, self.net.padding.height, self.net.padding.width)
        if tuple(image.shape) != shape or image.dtype != torch.float32:
            raise ValueError(f"expected padded float32 input {shape}, got {tuple(image.shape)} {image.dtype}")
        if self.net.skip != set(self.pruning["layer_indices"]):
            raise ValueError("the student's fixed whole-block pruning mask was modified")
        if self.pruning_spec is not None:
            validate_applied_spec(self.net, self.pruning_spec)
        outputs: dict[int, object] = {}
        int4.reset()
        with distillation_mode():
            for specs in self._groups:
                first, last = specs[0].index, specs[-1].index
                previous = outputs[first - 1] if first else None
                retained = tuple(spec.index for spec in specs
                                 if self.net.last_use[spec.index] > last or spec.index == last)
                if first in self.net.skip:
                    values = tuple(previous for _ in retained)
                else:
                    external = tuple(sorted({self.net.skip_source[spec.index] for spec in specs
                                             if self.net.skip_source[spec.index] is not None
                                             and self.net.skip_source[spec.index] < first}))
                    run = functools.partial(self._run_group, specs, external, retained)
                    args = (previous, image, *(outputs[index] for index in external))
                    if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                        values = checkpoint(run, *args, use_reentrant=False,
                                            context_fn=lambda: (distillation_mode(), distillation_mode()))
                    else:
                        values = run(*args)
                outputs.update(zip(retained, values))
                for index in [index for index in outputs
                              if self.net.last_use[index] <= last and index != last]:
                    del outputs[index]
        return _main(outputs[len(self.net.plan) - 1])[..., :self.net.height, :self.net.width]

    def export_state(self) -> dict[str, torch.Tensor]:
        """Materialize quantized weights under original NRNet keys, on the CPU.

        This neither removes parametrizations nor alters live master weights,
        optimizer state, gradients, training mode, or the fixed pruning mask.
        """
        state = {}
        with torch.no_grad():
            for key in self._canonical_keys:
                path, _, name = key.rpartition(".")
                module = self.net.get_submodule(path) if path else self.net
                state[key] = getattr(module, name).detach().cpu().clone().contiguous()
        return state
