"""Structure identity, safety boundaries, and costs of compact matrix shapes."""
from collections import Counter
from dataclasses import FrozenInstanceError, replace

import pytest

from dlssnr.ablation import layer_macs, units
from dlssnr.plan import build_plan
from dlssnr.pruning import PruningSpec, WidthSpec, layouts, validate_spec


def test_spec_normalizes_identity_without_reordering_mlp_groups():
    left = PruningSpec([6, 2], [WidthSpec(31, [[8, 4]], [3, 1]),
                                    WidthSpec(24, [[3, 1], [4, 2]])])
    right = PruningSpec((2, 6), (WidthSpec(24, ((1, 3), (2, 4))),
                                       WidthSpec(31, ((4, 8),), (1, 3))))
    assert left == right and hash(left) == hash(right)
    assert PruningSpec.from_dict(left.to_dict()) == left
    assert left.to_dict() == {
        "dropped_blocks": [2, 6],
        "widths": [{"block": 24, "mlp": [[1, 3], [2, 4]]},
                   {"block": 31, "mlp": [[4, 8]], "heads": [1, 3]}],
    }
    with pytest.raises(FrozenInstanceError):
        left.dropped_blocks = ()
    assert PruningSpec.from_dict({}) == PruningSpec()


@pytest.mark.parametrize("raw", [
    None, [], {"blocks": []}, {"dropped_blocks": None}, {"dropped_blocks": "1"},
    {"dropped_blocks": [True]}, {"dropped_blocks": [1.0]},
    {"dropped_blocks": [-1]}, {"dropped_blocks": [2, 2]},
    {"widths": None}, {"widths": {}}, {"widths": [None]},
    {"widths": [{"mlp": [[1]]}]}, {"widths": [{"block": 1}]},
    {"widths": [{"block": True, "heads": [0]}]},
    {"widths": [{"block": 1, "heads": [True]}]},
    {"widths": [{"block": 1, "heads": [0.0]}]},
    {"widths": [{"block": 1, "heads": [-1]}]},
    {"widths": [{"block": 1, "heads": []}]},
    {"widths": [{"block": 1, "heads": [0, 0]}]},
    {"widths": [{"block": 1, "heads": None}]},
    {"widths": [{"block": 1, "heads": [0], "other": 1}]},
    {"widths": [{"block": 1, "mlp": []}]},
    {"widths": [{"block": 1, "mlp": [[]]}]},
    {"widths": [{"block": 1, "mlp": [1]}]},
    {"widths": [{"block": 1, "mlp": [[0, 0]]}]},
    {"widths": [{"block": 1, "mlp": None, "heads": [0]}]},
    {"widths": [{"block": 1, "heads": [0]}, {"block": 1, "heads": [0]}]},
    {"dropped_blocks": [1], "widths": [{"block": 1, "heads": [0]}]},
])
def test_untrusted_spec_rejects_invalid_types_duplicates_and_overlap(raw):
    with pytest.raises(ValueError):
        PruningSpec.from_dict(raw)


def test_layouts_are_original_internal_dimensions_and_all_59_safe_blocks():
    plan = build_plan(1920, 1080)
    result = layouts(plan)
    assert Counter(layout.family for layout in result.values()) == {
        "swin": 36, "split": 15, "vit": 8}
    assert result.keys() == units(plan).keys()
    for block, layout in result.items():
        assert layout.block == block
        assert layout.indices == units(plan)[block].indices
        if layout.family == "swin":
            assert layout.hidden_size == 128
            assert layout.groups == layout.heads == layout.channels // 32
        elif layout.family == "split":
            assert (layout.groups, layout.hidden_size, layout.heads) == (8, 256, 16)
        else:
            assert (layout.groups, layout.hidden_size, layout.heads) == (1, 4096, 32)


def test_empty_and_explicit_full_indices_preserve_exact_full_cost():
    plan = build_plan(1920, 1080)
    full = PruningSpec(widths=tuple(WidthSpec(
        block, tuple(tuple(range(layout.hidden_size)) for _ in range(layout.groups)),
        tuple(range(layout.heads))) for block, layout in layouts(plan).items()))
    for spec in (PruningSpec(), full):
        result = validate_spec(plan, spec)
        assert result["total_macs"] == result["macs"] == 508_853_747_712
        assert result["keep_ratio"] == 1
        assert result["blocks"] == result["layer_indices"] == []


def test_depth_only_cost_matches_existing_accounting():
    plan = build_plan(1920, 1080)
    spec = PruningSpec((2, 6, 24, 31, 47))
    result = validate_spec(plan, spec)
    expected = sum(layer_macs(plan).values()) - sum(
        units(plan)[block].macs for block in spec.dropped_blocks)
    assert result["macs"] == expected
    assert result["layer_indices"] == sorted(
        layer.index for layer in plan if layer.block in spec.dropped_blocks)


@pytest.mark.parametrize("block", [0, 4, 8, 14, 22, 30, 39, 48, 56, 62, 66, 70, 999])
def test_both_depth_and_width_reject_protected_or_unknown_blocks(block):
    plan = build_plan(320, 320)
    for spec in (PruningSpec((block,)), PruningSpec(widths=(WidthSpec(block, heads=(0,)),))):
        with pytest.raises(ValueError, match="unsafe or unknown"):
            validate_spec(plan, spec)


@pytest.mark.parametrize("width,match", [
    (WidthSpec(6, (tuple(range(32)),)), "groups"),
    (WidthSpec(6, (tuple(range(32)), tuple(range(64)))), "equal"),
    (WidthSpec(6, (tuple(range(33)), tuple(range(33)))), "multiple of 32"),
    (WidthSpec(6, (tuple(range(128, 160)), tuple(range(32)))), "bounds"),
    (WidthSpec(6, heads=(2,)), "bounds"),
])
def test_dimensions_and_head_bounds_are_checked_against_original_plan(width, match):
    with pytest.raises(ValueError, match=match):
        validate_spec(build_plan(320, 320), PruningSpec(widths=(width,)))


@pytest.mark.parametrize("block", [2, 6, 10, 16])
def test_swin_mlp_and_head_costs_use_compact_shapes_on_shifted_tokens(block):
    layer = next(spec for spec in build_plan(1920, 1080) if spec.block == block)
    layout = layouts([layer])[block]
    groups, channels = layout.groups, layout.channels
    hidden = 96  # A valid non-grid index subset of the requested width.
    heads = max(1, layout.heads // 2)
    spec = PruningSpec(widths=(WidthSpec(
        block, tuple(tuple(range(32, 128)) for _ in range(groups)), tuple(range(heads))),))
    mlp = channels * groups * hidden + hidden * channels
    if groups > 1:
        mlp += channels * channels
    qkv = 3 * channels * heads * 32
    projection = heads * 32 * channels
    attention = 2 * 64 * heads * 32
    expected = layer.grid_x * layer.grid_y * 64 * (mlp + qkv + projection + attention)
    assert validate_spec([layer], spec)["macs"] == expected


def test_split_uses_raw_tokens_for_mlp_and_projection_but_padded_attention_tokens():
    plan = [spec for spec in build_plan(1920, 1080) if spec.block == 24]
    raw = plan[0].w * plan[0].h
    window = plan[2].grid_x * plan[2].grid_y * 64
    retained_heads = tuple(range(1, 16, 2))
    retained_mlp = tuple(tuple(range(64, 256)) for _ in range(8))
    spec = PruningSpec(widths=(WidthSpec(24, retained_mlp, retained_heads),))
    expected = raw * (512 * 512 + 2 * 8 * 64 * 192)  # Grouped MLP.
    expected += raw * 512 * 512  # Fixed output projection after MLP.
    expected += window * (3 * 512 * 256 + 2 * 64 * 256)
    expected += raw * 256 * 512  # Narrowed attention output projection.
    assert validate_spec(plan, spec)["macs"] == expected


def test_vit_uses_compact_matrices_with_original_padded_key_count():
    plan = [replace(spec, w=8, h=10, tokens=80, out_w=8, out_h=10)
            for spec in build_plan(1920, 1080) if spec.block == 31]
    spec = PruningSpec(widths=(WidthSpec(31, (tuple(range(1024, 4096)),),
                                                tuple(range(8, 32))),))
    expected = 2 * 80 * 1024 * 3072
    expected += 80 * (3 * 1024 * 768 + 768 * 1024)
    expected += 2 * 80 * 128 * 768
    assert validate_spec(plan, spec)["macs"] == expected


def test_width_and_depth_on_distinct_blocks_have_additive_savings():
    plan = build_plan(1920, 1080)
    full = validate_spec(plan, PruningSpec())["macs"]
    depth = PruningSpec((2, 31))
    width = PruningSpec(widths=(WidthSpec(24, heads=(0, 1, 3, 5)),))
    mixed = PruningSpec(depth.dropped_blocks, width.widths)
    assert full - validate_spec(plan, mixed)["macs"] == (
        full - validate_spec(plan, depth)["macs"] + full - validate_spec(plan, width)["macs"])
    assert validate_spec(plan, mixed)["layer_indices"] == validate_spec(plan, depth)["layer_indices"]


def test_1080_width_grid_reaches_eighty_percent_and_deeper_steps_reduce_cost():
    plan = build_plan(1920, 1080)
    candidates = layouts(plan)
    costs = []
    for fraction in (1, 0.75, 0.5):
        spec = PruningSpec(widths=tuple(WidthSpec(
            block, tuple(tuple(range(int(layout.hidden_size * fraction)))
                         for _ in range(layout.groups)),
            tuple(range(max(1, int(layout.heads * fraction)))))
            for block, layout in candidates.items()))
        costs.append(validate_spec(plan, spec)["macs"])
    assert costs[0] > costs[1] > costs[2]
    assert costs[2] <= 0.8 * costs[0]
    assert validate_spec(plan, spec, keep_ratio=0.8)["keep_ratio"] <= 0.8
    with pytest.raises(ValueError, match="exceeding"):
        validate_spec(plan, PruningSpec(), keep_ratio=0.8)


def test_tiny_vit_accepts_its_full_unaligned_hidden_width():
    plan = [replace(spec, channels=32, c_in=(10 if spec.layer == 1 else 32),
                    c_out=(10 if spec.layer == 0 else 32))
            for spec in build_plan(320, 320) if spec.block == 31]
    spec = PruningSpec(widths=(WidthSpec(31, (tuple(range(10)),), (0,)),))
    assert validate_spec(plan, spec)["macs"] == sum(layer_macs(plan).values())


@pytest.mark.parametrize("ratio", [0, -1, 1.1, float("nan"), float("inf"), True, None, "0.8"])
def test_invalid_ratio_rejected(ratio):
    with pytest.raises(ValueError, match="keep_ratio"):
        validate_spec(build_plan(320, 320), PruningSpec(), keep_ratio=ratio)


def test_empty_plan_and_non_spec_are_rejected():
    with pytest.raises(ValueError, match="positive MAC"):
        validate_spec([], PruningSpec())
    with pytest.raises(ValueError, match="PruningSpec"):
        validate_spec(build_plan(320, 320), {})
