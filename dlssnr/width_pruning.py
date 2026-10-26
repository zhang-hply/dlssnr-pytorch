"""Deterministic width ranking and physical slicing of the reference network.

Residual streams retain their original channel counts. Only paired MLP hidden
axes and complete 32-channel attention heads are narrowed. The reusable applier
keeps an immutable CPU source so search candidates never inherit previous cuts.
"""
from __future__ import annotations

import torch
from torch import nn

from . import int4, int8, quant
from .layers import split_swin, swin, vit
from .pruning import PruningSpec, layouts, validate_spec
from .runtime import clear_caches


def _require_buffers(net) -> None:
    if any(True for _ in net.parameters()):
        raise ValueError("apply structural pruning before promoting trainable parameters")
    if int4.active() or int8.sites or quant.skip_blocks or any(
            isinstance(module, (int4.Int4Matrix, int4.VitFfn, int4.SwinSite))
            for module in net.modules()):
        raise ValueError("structural pruning requires reference FP8 weights; disable INT4, INT8 and FP8-skip")


def _permutation(scores: torch.Tensor) -> tuple[int, ...]:
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite weights cannot be ranked for width pruning")
    values = scores.tolist()
    return tuple(sorted(range(len(values)), key=lambda index: (-values[index], index)))


def _double(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float64)


def rank_widths(net) -> dict[int, dict[str, tuple]]:
    """Rank every original hidden channel and head, highest importance first.

    Hidden scores are incoming-row L2 times outgoing-column L2. Head scores
    use V and the output projection; normalized Q/K magnitudes are not a useful
    scale measure. Each group is ranked independently, ties by original index.
    """
    _require_buffers(net)
    if getattr(net, "pruning_spec", PruningSpec()).widths:
        raise ValueError("rank widths on the original full-width network")
    rankings = {}
    for block, layout in layouts(net.plan).items():
        layers = [net.layers[index] for index in layout.indices]
        channels, groups, hidden, heads = (layout.channels, layout.groups,
                                            layout.hidden_size, layout.heads)
        if layout.family == "swin":
            params = layers[0].params
            incoming = _double(params.expand).reshape(groups, hidden, channels)
            outgoing = (_double(params.contract).unsqueeze(0) if groups == 1 else
                        _double(params.mid).reshape(groups, 32, hidden))
            value = _double(params.qkv).reshape(heads, 3, 32, channels)[:, 2]
            output = _double(params.out_proj).reshape(channels, heads, 32).permute(1, 0, 2)
        elif layout.family == "split":
            incoming = _double(layers[0].expand_hidden).transpose(1, 2)
            outgoing = _double(layers[0].contract).transpose(1, 2)
            value = _double(layers[2].qkv).t().reshape(heads, 3, 32, channels)[:, 2]
            output = _double(layers[3].weight).reshape(heads, 32, channels)
        else:
            incoming = _double(layers[0].matmul.weight).unsqueeze(0)
            outgoing = _double(layers[1].matmul.weight).unsqueeze(0)
            value = _double(layers[2].matmul.weight).reshape(3, heads, 32, channels)[2]
            output = _double(layers[4].matmul.weight).reshape(channels, heads, 32).permute(1, 0, 2)
        scores = torch.linalg.vector_norm(incoming, dim=-1) * torch.linalg.vector_norm(outgoing, dim=1)
        head_scores = (torch.linalg.vector_norm(value.flatten(1), dim=1)
                       * torch.linalg.vector_norm(output.flatten(1), dim=1))
        rankings[block] = {"mlp": tuple(_permutation(group) for group in scores),
                           "heads": _permutation(head_scores)}
    return rankings


def _select(value: torch.Tensor, dim: int, indices) -> torch.Tensor:
    index = torch.tensor(indices, device=value.device, dtype=torch.long)
    return value.index_select(dim, index).contiguous()


def _head_channels(heads) -> tuple[int, ...]:
    return tuple(head * 32 + channel for head in heads for channel in range(32))


def _slice_width(net, layout, width) -> None:
    layers = [net.layers[index] for index in layout.indices]
    channels, groups, hidden = layout.channels, layout.groups, layout.hidden_size
    if layout.family == "swin":
        params = layers[0].params
        if width.mlp is not None:
            selected = tuple(group * hidden + index for group, indexes in enumerate(width.mlp)
                             for index in indexes)
            params.expand = _select(params.expand, 0, selected)
            if groups == 1:
                params.contract = _select(params.contract, 1, width.mlp[0])
            else:
                middle = params.mid.reshape(groups, 32, hidden)
                params.mid = torch.stack([_select(middle[group], 1, indexes)
                                          for group, indexes in enumerate(width.mlp)]).reshape(
                                              channels, -1).contiguous()
        if width.heads is not None:
            qkv = params.qkv.reshape(layout.heads, 3, 32, channels)
            params.qkv = _select(qkv, 0, width.heads).reshape(-1, channels)
            params.out_proj = _select(params.out_proj, 1, _head_channels(width.heads))
            params.head_scale = _select(params.head_scale, 0, width.heads)
            params.exp_bias = _select(params.exp_bias, 0, width.heads)
            params.attention_heads = len(width.heads)
    elif layout.family == "split":
        ffn, _, attention, projection = layers
        if width.mlp is not None:
            ffn.expand_hidden = torch.stack([_select(ffn.expand_hidden[group], 1, indexes)
                                            for group, indexes in enumerate(width.mlp)])
            ffn.contract = torch.stack([_select(ffn.contract[group], 0, indexes)
                                       for group, indexes in enumerate(width.mlp)])
        if width.heads is not None:
            qkv = attention.qkv.reshape(channels, layout.heads, 3, 32)
            attention.qkv = _select(qkv, 1, width.heads).reshape(channels, -1)
            attention.bias = _select(attention.bias, 0, width.heads)
            attention.scale = _select(attention.scale, 0, width.heads)
            attention.heads = len(width.heads)
            projection.weight = _select(projection.weight, 0, _head_channels(width.heads))
    else:
        expand, contract, qkv, attention, projection = layers
        if width.mlp is not None:
            expand.matmul.weight = _select(expand.matmul.weight, 0, width.mlp[0])
            contract.matmul.weight = _select(contract.matmul.weight, 1, width.mlp[0])
        if width.heads is not None:
            selected = _head_channels(width.heads)
            qkv_indices = tuple(part * channels + index for part in range(3) for index in selected)
            qkv.matmul.weight = _select(qkv.matmul.weight, 0, qkv_indices)
            qkv.temperature = _select(qkv.temperature, 0, width.heads)
            qkv.channels = attention.channels = len(selected)
            projection.matmul.weight = _select(projection.matmul.weight, 1, selected)


class PruningApplier:
    """Reusable full-weight source for inference-only candidate evaluation."""

    def __init__(self, net):
        _require_buffers(net)
        if getattr(net, "pruning_spec", PruningSpec()).widths:
            raise ValueError("construct the pruning applier from original full-width weights")
        self.net = net
        self.layouts = layouts(net.plan)
        self._buffers = []
        self._attributes = []
        for module in net.modules():
            for name, value in module._buffers.items():
                if value is not None:
                    self._buffers.append((module, name, value.detach().cpu().clone()))
            if isinstance(module, swin.SwinParams):
                self._attributes.extend((module, name, getattr(module, name))
                                        for name in ("channels", "heads", "attention_heads"))
            elif isinstance(module, split_swin.SplitQKVAttn):
                self._attributes.append((module, "heads", module.heads))
            elif isinstance(module, (vit.VitQKV, vit.VitAttention)):
                self._attributes.append((module, "channels", module.channels))

    def apply(self, spec: PruningSpec):
        """Restore the source, apply a validated candidate and clear weight caches."""
        _require_buffers(self.net)
        stats = validate_spec(self.net.plan, spec, keep_ratio=1.0)
        with torch.no_grad():
            for module, name, original in self._buffers:
                module._buffers[name] = original.to(device=module._buffers[name].device, copy=True)
            for module, name, value in self._attributes:
                setattr(module, name, value)
            for width in spec.widths:
                _slice_width(self.net, self.layouts[width.block], width)
        self.net.skip = set(stats["layer_indices"])
        self.net.pruning_spec = spec
        clear_caches()
        return self.net

    def reset(self):
        return self.apply(PruningSpec())


def apply_spec(net, spec: PruningSpec):
    """Apply once to a fresh buffer network, before loading or promoting weights."""
    return PruningApplier(net).apply(spec)


def _shape(module: nn.Module, name: str) -> tuple[int, ...]:
    # Avoid evaluating every FP8 parametrization merely to check its dimensions.
    parametrizations = getattr(module, "parametrizations", None)
    value = (parametrizations[name].original if parametrizations is not None and name in parametrizations
             else getattr(module, name))
    return tuple(value.shape)


def validate_applied_spec(net, spec: PruningSpec) -> None:
    """Detect changed masks, runtime head counts or narrowed master shapes."""
    stats = validate_spec(net.plan, spec, keep_ratio=1.0)
    if getattr(net, "pruning_spec", None) != spec or net.skip != set(stats["layer_indices"]):
        raise ValueError("the student's fixed pruning specification was modified")
    available = layouts(net.plan)
    for width in spec.widths:
        layout = available[width.block]
        layers = [net.layers[index] for index in layout.indices]
        c, g = layout.channels, layout.groups
        k = layout.hidden_size if width.mlp is None else len(width.mlp[0])
        h = layout.heads if width.heads is None else len(width.heads)
        a = h * 32
        if layout.family == "swin":
            p = layers[0].params
            correct = (p.heads == g and p.channels == c and p.attention_heads == h
                       and _shape(p, "expand") == (g * k, c)
                       and _shape(p, "contract") == (c, k if g == 1 else c)
                       and (p.mid is None if g == 1 else _shape(p, "mid") == (c, k))
                       and _shape(p, "qkv") == (3 * a, c) and _shape(p, "out_proj") == (c, a)
                       and _shape(p, "head_scale") == (h,) and _shape(p, "exp_bias") == (h, 64, 64))
        elif layout.family == "split":
            ffn, _, attn, projection = layers
            correct = (_shape(ffn, "expand_hidden") == (g, 64, k)
                       and _shape(ffn, "contract") == (g, k, 64) and attn.heads == h
                       and _shape(attn, "qkv") == (c, 3 * a)
                       and _shape(attn, "scale") == (h,) and _shape(attn, "bias") == (h, 64, 64)
                       and _shape(projection, "weight") == (a, c))
        else:
            expand, contract, qkv, attention, projection = layers
            correct = (_shape(expand.matmul, "weight") == (k, c)
                       and _shape(contract.matmul, "weight") == (c, k)
                       and _shape(qkv.matmul, "weight") == (3 * a, c)
                       and _shape(qkv, "temperature") == (h,)
                       and qkv.channels == attention.channels == a
                       and _shape(projection.matmul, "weight") == (c, a))
        if not correct:
            raise ValueError(f"the student's fixed width structure was modified at block {width.block}")
