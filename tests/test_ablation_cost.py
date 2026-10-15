from collections import Counter
from dataclasses import replace

import pytest

from dlssnr.ablation import layer_macs, units
from dlssnr.plan import build_plan


def _by_block(width=1920, height=1080):
    return {(spec.block, spec.layer): spec for spec in build_plan(width, height)}


@pytest.mark.parametrize("heads,block", [(1, 2), (2, 6), (4, 10), (8, 16)])
def test_swin_matrix_shapes_and_shifted_windows(heads, block):
    spec = _by_block()[block, 0]
    channels = spec.channels
    tokens = spec.grid_x * spec.grid_y * 64
    mlp = channels * 4 * channels
    if heads > 1:
        mlp += heads * (4 * channels // heads) * (channels // heads)
        mlp += channels * channels
    else:
        mlp += 4 * channels * channels
    qkv_and_projection = channels * 3 * channels + channels * channels
    attention = 2 * 64 * channels
    assert layer_macs([spec])[spec.index] == tokens * (mlp + qkv_and_projection + attention)
    assert tokens > spec.w * spec.h


def test_split_matrices_use_raw_extent_except_attention():
    specs = [_by_block()[24, layer] for layer in range(4)]
    costs = layer_macs(specs)
    raw = specs[0].w * specs[0].h
    assert costs[specs[0].index] == raw * 8 * (512 * 64 + 64 * 256 + 256 * 64)
    assert costs[specs[1].index] == raw * 512 * 512
    attn = specs[2]
    assert costs[attn.index] == attn.grid_x * attn.grid_y * (64 * 512 * 1536 + 2 * 16 * 64 * 64 * 32)
    assert costs[specs[3].index] == raw * 512 * 512


def test_vit_counts_each_projection_once_and_pads_keys():
    # An extent with 80 queries exposes the attention's 64-key alignment.
    specs = [replace(_by_block()[31, layer], w=8, h=10, tokens=80, out_w=8, out_h=10)
             for layer in range(5)]
    costs = layer_macs(specs)
    assert [costs[spec.index] for spec in specs] == [
        80 * 1024 * 4096,
        80 * 4096 * 1024,
        80 * 1024 * 3072,
        2 * 80 * 128 * 1024,
        80 * 1024 * 1024,
    ]


def test_pre_post_use_full_resolution_32_channel_body():
    plan = build_plan(1920, 1080)
    pre, post = plan[0], plan[-1]
    costs = layer_macs(plan)
    body_per_token = 32 * 128 + 128 * 32 + 32 * 96 + 32 * 32 + 2 * 64 * 32
    assert costs[pre.index] == pre.w * pre.h * 16 * 32 + pre.grid_x * pre.grid_y * 64 * body_per_token
    assert costs[post.index] == post.grid_x * post.grid_y * 64 * body_per_token + post.out_w * post.out_h * 32 * 3
    assert costs[pre.index] == 35_295_068_160
    assert costs[post.index] == 34_821_373_952


def test_resampling_projections_use_extent_before_padding_or_replication():
    plan = build_plan(1920, 1080)
    specs = {(spec.block, spec.layer): spec for spec in plan}
    costs = layer_macs(plan)
    down = specs[22, 0]
    body_only = replace(down, variant="")
    assert costs[down.index] - layer_macs([body_only])[down.index] == (down.w // 2) * (down.h // 2) * 256 * 512
    # The output is padded after the projection, so charging its 36x60 extent
    # instead of the actual 34x60 pooled extent would overcount this layer.
    assert (down.w // 2, down.h // 2) != (down.out_w, down.out_h)
    up = specs[48, 0]
    assert costs[up.index] - layer_macs([replace(up, variant="")])[up.index] == up.w * up.h * 512 * 256
    decoder, previous = specs[39, 0], specs[38, 4]
    assert costs[decoder.index] == previous.out_w * previous.out_h * 1024 * 512
    assert (decoder.w, decoder.h) != (previous.out_w, previous.out_h)
    final = specs[30, 4]
    assert costs[final.index] == final.w * final.h * 512 * 1024


@pytest.mark.parametrize("size,expected", [
    ((1920, 1080), 508_853_747_712),
    ((3840, 2160), 1_991_052_361_728),
])
def test_total_mac_regression(size, expected):
    plan = build_plan(*size)
    costs = layer_macs(plan)
    assert set(costs) == {spec.index for spec in plan}
    assert sum(costs.values()) == expected


@pytest.mark.parametrize("size", [(1920, 1080), (100, 50), (1366, 768), (3840, 2160)])
def test_all_59_safe_blocks_include_layout_views(size):
    plan = build_plan(*size)
    candidates = units(plan)
    costs = layer_macs(plan)
    assert Counter(unit.family for unit in candidates.values()) == {"swin": 36, "split": 15, "vit": 8}
    assert set(candidates) == set(range(1, 70)) - {4, 8, 14, 22, 30, 39, 48, 56, 62, 66}
    assert {1, 5, 9, 15, 23, 47, 55, 61, 65, 69} <= candidates.keys()
    for block, unit in candidates.items():
        assert unit.block == block
        assert unit.indices == tuple(spec.index for spec in plan if spec.block == block)
        assert unit.macs == sum(costs[index] for index in unit.indices)
        assert all((spec.w, spec.h) == (spec.out_w, spec.out_h) for spec in plan if spec.block == block)


def test_partial_or_shape_changing_block_is_not_a_candidate():
    plan = build_plan(1920, 1080)
    partial = [spec for spec in plan if not (spec.block == 31 and spec.layer == 4)]
    assert 31 not in units(partial)
    altered = [replace(spec, out_w=spec.out_w + 1) if spec.block == 2 else spec for spec in plan]
    assert 2 not in units(altered)


def test_unknown_layers_and_duplicate_indices_are_rejected():
    pre = build_plan(1920, 1080)[0]
    with pytest.raises(ValueError, match="unsupported layer kind"):
        layer_macs([replace(pre, kind="Unknown")])
    with pytest.raises(ValueError, match="duplicate layer index"):
        layer_macs([pre, pre])
