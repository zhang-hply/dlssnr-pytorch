"""Weight-independent MAC accounting and safe whole-block ablation candidates.

One matrix multiply-accumulate counts as one MAC. These estimates include learned
projections and the two attention matrix products, including their padded tokens.
They exclude elementwise arithmetic, reductions, padding, and data movement; they
are a computation budget, not an estimate of elapsed inference time.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .plan import LayerSpec


@dataclass(frozen=True)
class BlockUnit:
    """A whole block whose layers can all be bypassed without changing shape."""

    block: int
    family: str
    indices: tuple[int, ...]
    macs: int


def _swin_body(tokens: int, channels: int, heads: int) -> int:
    # A single head expands C -> 4C -> C. Multiple heads add the grouped
    # 4C -> C middle projection, then use a dense C -> C contraction.
    mlp = (8 * channels * channels if heads == 1 else
           5 * channels * channels + 4 * channels * channels // heads)
    projections = 4 * channels * channels  # QKV and attention output
    attention = 2 * 64 * channels         # Q K^T and P V per window token
    return tokens * (mlp + projections + attention)


def layer_macs(plan: Sequence[LayerSpec]) -> dict[int, int]:
    """Return matrix MACs by layer index for a network plan in execution order.

    Swin bodies operate on shifted 8x8 windows, while split FFNs/projections
    operate on the unpadded feature map. The pre/post bodies always use C=32,
    even though their plan records describe RGB input/output channels. Decoder
    upsampling projects the preceding, lower-resolution tensor before replication.
    """
    costs: dict[int, int] = {}
    previous: LayerSpec | None = None
    for spec in plan:
        raw_tokens = spec.w * spec.h
        window_tokens = spec.grid_x * spec.grid_y * 64
        channels = spec.channels
        kind = spec.kind
        if kind == "PreBlock":
            cost = raw_tokens * 16 * 32 + _swin_body(window_tokens, 32, 1)
        elif kind == "PostBlock":
            cost = _swin_body(window_tokens, 32, 1) + spec.out_w * spec.out_h * 32 * 3
        elif kind in {"Swin1H", "Swin2H", "Swin4H", "Swin8H"}:
            cost = _swin_body(window_tokens, channels, spec.heads)
            if spec.variant == "ds":
                # The projection reads the pooled map before output padding.
                cost += (spec.w // 2) * (spec.h // 2) * spec.c_in * spec.c_out
            elif spec.variant == "upsample":
                cost += raw_tokens * spec.c_in * spec.c_out
        elif kind == "SplitFfwd":
            # Eight groups: 512 -> 64 -> 256 -> 64, then concatenate.
            cost = raw_tokens * (channels * channels + 2 * 8 * 64 * 256)
        elif kind in {"SplitFfwdProj", "SplitProj", "SplitProjPool", "SplitFinalHead"}:
            cost = raw_tokens * spec.c_in * spec.c_out
        elif kind == "SplitQKVAttn":
            cost = window_tokens * (3 * channels * channels + 2 * 64 * channels)
        elif kind in {"VitFfnExpand", "VitFfnContract", "VitProjection"}:
            cost = raw_tokens * spec.c_in * spec.c_out
        elif kind == "VitQKV":
            cost = raw_tokens * 3 * channels * channels
        elif kind == "VitAttention":
            padded_keys = (raw_tokens + 63) // 64 * 64
            cost = 2 * raw_tokens * padded_keys * channels
        elif kind == "DecInputUpsample":
            if previous is None:
                raise ValueError("DecInputUpsample needs the preceding layer's output extent")
            cost = previous.out_w * previous.out_h * spec.c_in * spec.c_out
        else:
            raise ValueError(f"unsupported layer kind: {kind}")
        if spec.index in costs:
            raise ValueError(f"duplicate layer index: {spec.index}")
        costs[spec.index] = cost
        previous = spec
    return costs


def units(plan: Sequence[LayerSpec]) -> dict[int, BlockUnit]:
    """Return all safe whole-block candidates, keyed by block number.

    Layout-only inpview/outview variants remain eligible. Sampling blocks, the
    pre/post blocks, and the split block containing the final head are excluded.
    Partial split/ViT blocks are never exposed as whole-block candidates.
    """
    costs = layer_macs(plan)
    grouped: dict[int, list[LayerSpec]] = {}
    for spec in plan:
        grouped.setdefault(spec.block, []).append(spec)

    split = ("SplitFfwd", "SplitFfwdProj", "SplitQKVAttn", "SplitProj")
    vit = ("VitFfnExpand", "VitFfnContract", "VitQKV", "VitAttention", "VitProjection")
    candidates: dict[int, BlockUnit] = {}
    for block, specs in grouped.items():
        kinds = tuple(spec.kind for spec in specs)
        first = specs[0]
        if (len(specs) == 1 and first.kind in {"Swin1H", "Swin2H", "Swin4H", "Swin8H"}
                and first.variant in {"", "inpview", "outview"}
                and first.c_in == first.c_out):
            family = "swin"
        elif kinds == split:
            family = "split"
        elif kinds == vit:
            family = "vit"
        else:
            continue
        if any((spec.w, spec.h) != (spec.out_w, spec.out_h) for spec in specs):
            continue
        indices = tuple(spec.index for spec in specs)
        candidates[block] = BlockUnit(block, family, indices, sum(costs[index] for index in indices))
    return candidates
