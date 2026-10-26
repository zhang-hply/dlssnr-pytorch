"""Canonical depth/width pruning specifications and matrix-MAC accounting.

Widths describe retained indices in the *original* model, never indices in an
already pruned model. They change only internal MLP and attention dimensions;
block inputs, outputs, and residual connections keep their original widths.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .ablation import layer_macs, units
from .plan import LayerSpec


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _indices(values: object, name: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of integer indices")
    result = tuple(_integer(value, name) for value in values)
    if not result and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate indices")
    return tuple(sorted(result))


@dataclass(frozen=True)
class WidthSpec:
    """Retained original MLP indices per group and/or attention head indices."""

    block: int
    mlp: tuple[tuple[int, ...], ...] | None = None
    heads: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _integer(self.block, "block")
        if self.mlp is None and self.heads is None:
            raise ValueError("a width specification needs mlp and/or heads indices")
        if self.mlp is not None:
            if not isinstance(self.mlp, (list, tuple)) or not self.mlp:
                raise ValueError("mlp must be a nonempty list or tuple of index groups")
            object.__setattr__(self, "mlp", tuple(
                _indices(group, "mlp group") for group in self.mlp))
        if self.heads is not None:
            object.__setattr__(self, "heads", _indices(self.heads, "heads"))

    def to_dict(self) -> dict:
        result: dict = {"block": self.block}
        if self.mlp is not None:
            result["mlp"] = [list(group) for group in self.mlp]
        if self.heads is not None:
            result["heads"] = list(self.heads)
        return result

    @classmethod
    def from_dict(cls, raw: object) -> WidthSpec:
        if not isinstance(raw, dict):
            raise ValueError("a width specification must be an object")
        if set(raw) - {"block", "mlp", "heads"} or "block" not in raw:
            raise ValueError("width fields must be block and optional mlp/heads")
        if ("mlp" in raw and raw["mlp"] is None) or ("heads" in raw and raw["heads"] is None):
            raise ValueError("omit unused mlp/heads fields rather than setting them to null")
        return cls(block=raw["block"], mlp=raw.get("mlp"), heads=raw.get("heads"))


@dataclass(frozen=True)
class PruningSpec:
    """Hashable, order-independent identity of one depth/width structure."""

    dropped_blocks: tuple[int, ...] = ()
    widths: tuple[WidthSpec, ...] = ()

    def __post_init__(self) -> None:
        dropped = _indices(self.dropped_blocks, "dropped_blocks", allow_empty=True)
        if not isinstance(self.widths, (list, tuple)) or any(
                not isinstance(width, WidthSpec) for width in self.widths):
            raise ValueError("widths must be a list or tuple of WidthSpec objects")
        widths = tuple(sorted(self.widths, key=lambda width: width.block))
        blocks = [width.block for width in widths]
        if len(blocks) != len(set(blocks)):
            raise ValueError("widths must not contain duplicate blocks")
        if set(blocks) & set(dropped):
            raise ValueError("a dropped block cannot also have a width specification")
        object.__setattr__(self, "dropped_blocks", dropped)
        object.__setattr__(self, "widths", widths)

    def to_dict(self) -> dict:
        return {"dropped_blocks": list(self.dropped_blocks),
                "widths": [width.to_dict() for width in self.widths]}

    @classmethod
    def from_dict(cls, raw: object) -> PruningSpec:
        if not isinstance(raw, dict) or set(raw) - {"dropped_blocks", "widths"}:
            raise ValueError("pruning specification fields must be dropped_blocks and widths")
        widths = raw.get("widths", [])
        if not isinstance(widths, (list, tuple)):
            raise ValueError("widths must be a list of width specifications")
        return cls(dropped_blocks=raw.get("dropped_blocks", ()),
                   widths=tuple(WidthSpec.from_dict(width) for width in widths))


@dataclass(frozen=True)
class BlockLayout:
    """Original dimensions of a safe block; hidden_size is per MLP group."""

    block: int
    family: str
    indices: tuple[int, ...]
    channels: int
    groups: int
    hidden_size: int
    heads: int


def layouts(plan: Sequence[LayerSpec]) -> dict[int, BlockLayout]:
    """Describe the same safe blocks exposed by whole-block ablation."""
    by_index = {layer.index: layer for layer in plan}
    result = {}
    for block, unit in units(plan).items():
        first = by_index[unit.indices[0]]
        if unit.family == "swin":
            groups, heads = first.heads, first.heads
            hidden = 4 * first.channels // groups
        elif unit.family == "split":
            groups, heads, hidden = 8, first.channels // 32, 256
        else:
            groups, heads, hidden = 1, first.channels // 32, first.c_out
        result[block] = BlockLayout(block, unit.family, unit.indices, first.channels,
                                    groups, hidden, heads)
    return result


def _validate_width(layout: BlockLayout, width: WidthSpec) -> None:
    if width.mlp is not None:
        if len(width.mlp) != layout.groups:
            raise ValueError(f"block {width.block}: mlp needs {layout.groups} groups")
        sizes = {len(group) for group in width.mlp}
        if len(sizes) != 1:
            raise ValueError(f"block {width.block}: mlp groups must have equal hidden widths")
        size = len(width.mlp[0])
        if size != layout.hidden_size and size % 32:
            raise ValueError(f"block {width.block}: pruned mlp width must be a multiple of 32")
        if any(index >= layout.hidden_size for group in width.mlp for index in group):
            raise ValueError(f"block {width.block}: mlp index out of bounds")
    if width.heads is not None and any(index >= layout.heads for index in width.heads):
        raise ValueError(f"block {width.block}: head index out of bounds")


def _width_macs(layout: BlockLayout, width: WidthSpec,
                layers: dict[int, LayerSpec], original: dict[int, int]) -> int:
    """Recompute only affected matrix dimensions, retaining all other costs."""
    specs = [layers[index] for index in layout.indices]
    channels = layout.channels
    hidden = len(width.mlp[0]) if width.mlp is not None else layout.hidden_size
    # The original dimension is used for unmodified axes as well as full-index
    # specifications, preserving exact baseline costs even for tiny test plans.
    heads = len(width.heads) if width.heads is not None else layout.heads
    attention_width = channels if heads == layout.heads else heads * 32
    if layout.family == "swin":
        first = specs[0]
        tokens = first.grid_x * first.grid_y * 64
        mlp = channels * hidden * layout.groups
        if layout.groups == 1:
            mlp += hidden * channels
        else:
            mlp += hidden * channels + channels * channels
        attention = 4 * channels * attention_width + 2 * 64 * attention_width
        return tokens * (mlp + attention)

    costs = {layer.index: original[layer.index] for layer in specs}
    for layer in specs:
        raw_tokens = layer.w * layer.h
        if layer.kind == "SplitFfwd" and hidden != layout.hidden_size:
            # The original grouped input projection remains C -> 8 x 64;
            # only the two 64 <-> hidden matrices per group are narrowed.
            costs[layer.index] = raw_tokens * (channels * channels + 2 * 8 * 64 * hidden)
        elif layer.kind == "SplitQKVAttn" and attention_width != channels:
            tokens = layer.grid_x * layer.grid_y * 64
            costs[layer.index] = tokens * (3 * channels * attention_width
                                           + 2 * 64 * attention_width)
        elif layer.kind == "SplitProj" and attention_width != channels:
            costs[layer.index] = raw_tokens * attention_width * layer.c_out
        elif layer.kind == "VitFfnExpand" and hidden != layout.hidden_size:
            costs[layer.index] = raw_tokens * layer.c_in * hidden
        elif layer.kind == "VitFfnContract" and hidden != layout.hidden_size:
            costs[layer.index] = raw_tokens * hidden * layer.c_out
        elif layer.kind == "VitQKV" and attention_width != channels:
            costs[layer.index] = raw_tokens * 3 * channels * attention_width
        elif layer.kind == "VitAttention" and attention_width != channels:
            padded_keys = (raw_tokens + 63) // 64 * 64
            costs[layer.index] = 2 * raw_tokens * padded_keys * attention_width
        elif layer.kind == "VitProjection" and attention_width != channels:
            costs[layer.index] = raw_tokens * attention_width * layer.c_out
    return sum(costs.values())


def validate_spec(plan: Sequence[LayerSpec], spec: PruningSpec,
                  keep_ratio: float = 1.0) -> dict:
    """Validate structure and recompute its remaining MACs from the original plan.

    Only whole-block drops appear in ``layer_indices``. Width changes must also
    be applied to the layer matrices by the structural pruning implementation.
    The requested ratio is an upper bound; the returned ratio is the actual cost.
    """
    if (isinstance(keep_ratio, bool) or not isinstance(keep_ratio, (float, int))
            or not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1):
        raise ValueError("keep_ratio must be finite and within (0, 1]")
    if not isinstance(spec, PruningSpec):
        raise ValueError("spec must be a PruningSpec")
    candidates = layouts(plan)
    requested = set(spec.dropped_blocks) | {width.block for width in spec.widths}
    unsafe = sorted(requested - candidates.keys())
    if unsafe:
        raise ValueError(f"unsafe or unknown pruning candidates: {unsafe}")
    costs = layer_macs(plan)
    total = sum(costs.values())
    if total <= 0:
        raise ValueError("the model plan must have a positive MAC count")
    remaining = total
    for block in spec.dropped_blocks:
        remaining -= sum(costs[index] for index in candidates[block].indices)
    layers = {layer.index: layer for layer in plan}
    for width in spec.widths:
        layout = candidates[width.block]
        _validate_width(layout, width)
        original = sum(costs[index] for index in layout.indices)
        remaining += _width_macs(layout, width, layers, costs) - original
    if remaining > total * keep_ratio + 1e-6:
        raise ValueError(f"pruned model retains {remaining / total:.6%} MACs, "
                         f"exceeding the {keep_ratio:.6%} budget")
    return {"blocks": list(spec.dropped_blocks),
            "layer_indices": sorted(index for block in spec.dropped_blocks
                                    for index in candidates[block].indices),
            "total_macs": total, "macs": remaining, "keep_ratio": remaining / total,
            "spec": spec.to_dict()}
