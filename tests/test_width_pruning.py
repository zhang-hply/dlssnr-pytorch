"""Physical width cuts, independent full-width references and source restoration."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from dlssnr import quant
from dlssnr.layers import split_swin, swin, vit
from dlssnr.plan import build_plan
from dlssnr.pruning import PruningSpec, WidthSpec, layouts
from dlssnr.training import distillation_mode
from dlssnr.width_pruning import (PruningApplier, apply_spec, rank_widths,
                                  validate_applied_spec)
from test_distillation_student import _SmallWeights


def _block(family):
    """Small spatial extents with real kernels; ViT has a 64-wide residual."""
    source = build_plan(320, 320)
    block = {"swin1": 2, "swin": 6, "split": 24, "vit": 31}[family]
    specs = [replace(spec, index=index, w=2, h=3, out_w=2, out_h=3,
                     grid_x=1, grid_y=1, shift_x=0, shift_y=0, shifted=False)
             for index, spec in enumerate(spec for spec in source if spec.block == block)]
    generator = torch.Generator().manual_seed(241)

    def random(*shape):
        return quant.q8(torch.randn(*shape, generator=generator) * 0.03)

    def empty(cls):
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        return module

    if family.startswith("swin"):
        c, heads = specs[0].channels, specs[0].heads
        layer = empty(swin.SwinLayer)
        layer.spec = specs[0]
        layer.params = swin.SwinParams(
            c, heads, expand=random(4 * c, c),
            mid=random(c, 128) if heads > 1 else None,
            contract=random(c, c if heads > 1 else 4 * c),
            qkv=random(3 * c, c), out_proj=random(c, c),
            exp_bias=torch.full((heads, 64, 64), swin.EXP_OFFSET),
            head_scale=torch.full((heads,), 1.5),
            mlp_scale=torch.full((c,), 0.875), attn_scale=torch.full((c,), 0.75))
        layer.params.i4 = None
        layer.register_buffer("resample", None)
        layer.register_buffer("gain", None)
        layers = [layer]
    elif family == "split":
        ffn = empty(split_swin.SplitFfwd)
        ffn.register_buffer("expand_in", random(512, 512))
        ffn.register_buffer("expand_hidden", random(8, 64, 256))
        ffn.register_buffer("contract", random(8, 256, 64))
        projection = empty(split_swin.SplitFfwdProj)
        projection.block, projection.stash = block, False
        projection.register_buffer("weight", random(512, 512))
        projection.register_buffer("gain", torch.full((512,), 0.875))
        attention = empty(split_swin.SplitQKVAttn)
        attention.block, attention.i4 = block, None
        attention.shift_x = attention.shift_y = 0
        attention.register_buffer("qkv", random(512, 1536))
        attention.register_buffer("bias", torch.zeros(16, 64, 64))
        attention.register_buffer("scale", torch.full((16,), 1.5))
        out = empty(split_swin.SplitProj)
        out.register_buffer("weight", random(512, 512))
        out.register_buffer("gain", torch.full((512,), 0.75))
        layers = [ffn, projection, attention, out]
    else:
        dims = [(64, 256), (256, 64), (64, 64), (64, 64), (64, 64)]
        specs = [replace(spec, channels=64, c_in=cin, c_out=cout)
                 for spec, (cin, cout) in zip(specs, dims)]
        weights = _SmallWeights()
        layers = [vit.BUILDERS[spec.kind](spec, weights) for spec in specs]
    net = nn.Module()
    net.plan, net.layers, net.skip = specs, nn.ModuleList(layers), set()
    return net


def _case(net, *, mlp=True, heads=True):
    layout = next(iter(layouts(net.plan).values()))
    # Different selections in each MLP group detect wrong per-group gathering.
    groups = tuple(tuple(range(group % 2, layout.hidden_size, 2))
                   for group in range(layout.groups)) if mlp else None
    selected_heads = tuple(range(1, layout.heads, 2)) if heads and layout.heads > 1 else None
    return PruningSpec(widths=(WidthSpec(layout.block, mlp=groups, heads=selected_heads),))


def _run(net, x):
    family = next(iter(layouts(net.plan).values())).family
    if family == "swin":
        return net.layers[0](x)
    if family == "split":
        ffn, project, attention, out = net.layers
        residual = project(ffn(x))
        return out(attention(residual), residual)
    expand, contract, qkv, attention, project = net.layers
    residual = contract(expand(x), x)
    return project(attention(qkv(residual)), residual)


def _masked_reference(net, spec):
    """Keep original shapes and zero excluded consumers, independently of slicing."""
    width = spec.widths[0]
    layout = next(iter(layouts(net.plan).values()))
    family, c = layout.family, layout.channels
    if width.mlp is not None:
        for group, keep in enumerate(width.mlp):
            drop = sorted(set(range(layout.hidden_size)) - set(keep))
            if family == "swin":
                params = net.layers[0].params
                target = params.contract if layout.groups == 1 else params.mid[group * 32:(group + 1) * 32]
                target[:, drop] = 0
            elif family == "split":
                net.layers[0].contract[group, drop, :] = 0
            else:
                net.layers[1].matmul.weight[:, drop] = 0
    if width.heads is not None:
        drop = [head * 32 + dim for head in range(layout.heads) if head not in width.heads
                for dim in range(32)]
        if family == "swin":
            net.layers[0].params.out_proj[:, drop] = 0
        elif family == "split":
            net.layers[3].weight[drop, :] = 0
        else:
            net.layers[4].matmul.weight[:, drop] = 0
    return net


@pytest.mark.parametrize("family", ["swin1", "swin", "split", "vit"])
def test_full_width_and_reset_preserve_original_bits(family):
    net = _block(family)
    layout = next(iter(layouts(net.plan).values()))
    x = quant.q8(torch.randn(1, layout.channels, 2, 3, generator=torch.Generator().manual_seed(8)))
    original = {key: value.clone() for key, value in net.state_dict().items()}
    expected = _run(net, x)
    applier = PruningApplier(net)
    full_indices = PruningSpec(widths=(WidthSpec(
        layout.block, tuple(tuple(range(layout.hidden_size)) for _ in range(layout.groups)),
        tuple(range(layout.heads))),))
    applier.apply(full_indices)
    assert torch.equal(_run(net, x), expected)
    validate_applied_spec(net, full_indices)
    applier.apply(_case(net, heads=family != "swin1"))
    applier.reset()
    assert net.pruning_spec == PruningSpec() and net.skip == set()
    assert torch.equal(_run(net, x), expected)
    assert all(torch.equal(net.state_dict()[key], value) for key, value in original.items())


@pytest.mark.parametrize("family", ["swin", "split", "vit"])
@pytest.mark.parametrize("axes", ["mlp", "heads", "both"])
def test_width_cut_matches_independent_zero_mask_reference(family, axes):
    original = _block(family)
    spec = _case(original, mlp=axes != "heads", heads=axes != "mlp")
    pruned = apply_spec(copy.deepcopy(original), spec)
    masked = _masked_reference(copy.deepcopy(original), spec)
    channels = next(iter(layouts(original.plan).values())).channels
    x = quant.q8(torch.randn(1, channels, 2, 3, generator=torch.Generator().manual_seed(10)))
    actual, expected = _run(pruned, x), _run(masked, x)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    validate_applied_spec(pruned, spec)
    if family == "swin" and axes != "mlp":
        assert pruned.layers[0].params.heads == 2  # original rounding and MLP groups
        assert pruned.layers[0].params.attention_heads == 1


@pytest.mark.parametrize("family", ["swin", "split", "vit"])
def test_candidates_restore_original_indices_and_clear_caches(family, monkeypatch):
    net = _block(family)
    pristine = copy.deepcopy(net)
    calls = []
    monkeypatch.setattr("dlssnr.width_pruning.clear_caches", lambda: calls.append(1))
    applier = PruningApplier(net)
    first = _case(net)
    layout = next(iter(layouts(net.plan).values()))
    second = PruningSpec(widths=(WidthSpec(layout.block,
        tuple(tuple(range(layout.hidden_size // 4, layout.hidden_size)) for _ in range(layout.groups))),))
    applier.apply(first)
    applier.apply(second)
    expected = apply_spec(pristine, second)
    assert net.state_dict().keys() == expected.state_dict().keys()
    assert all(torch.equal(value, expected.state_dict()[name]) for name, value in net.state_dict().items())
    applier.apply(PruningSpec(dropped_blocks=(layout.block,)))
    assert net.skip == set(layout.indices)
    applier.reset()
    assert not net.skip and len(calls) == 5


@pytest.mark.parametrize("family", ["swin", "split", "vit"])
def test_rankings_use_paired_weights_and_original_index_ties(family):
    net = _block(family)
    layout = next(iter(layouts(net.plan).values()))
    c = layout.channels
    if family == "swin":
        p = net.layers[0].params
        incoming = p.expand.reshape(layout.groups, layout.hidden_size, c)
        outgoing = p.mid.reshape(layout.groups, 32, layout.hidden_size)
        value = p.qkv.reshape(layout.heads, 3, 32, c)[:, 2]
        output = p.out_proj
    elif family == "split":
        incoming = net.layers[0].expand_hidden.transpose(1, 2)
        outgoing = net.layers[0].contract.transpose(1, 2)
        value = net.layers[2].qkv.t().reshape(layout.heads, 3, 32, c)[:, 2]
        output = net.layers[3].weight
    else:
        incoming = net.layers[0].matmul.weight.unsqueeze(0)
        outgoing = net.layers[1].matmul.weight.unsqueeze(0)
        value = net.layers[2].matmul.weight.reshape(3, layout.heads, 32, c)[2]
        output = net.layers[4].matmul.weight
    incoming.zero_()
    outgoing.fill_(1)
    for group in range(layout.groups):
        incoming[group, 3 + group] = 2
        incoming[group, 19 + group] = 2
    value.fill_(1)
    value[-1].fill_(2)
    output.fill_(1)
    result = rank_widths(net)[layout.block]
    assert result == rank_widths(net)[layout.block]
    assert result["heads"] == (layout.heads - 1, *range(layout.heads - 1))
    for group, ranks in enumerate(result["mlp"]):
        assert ranks[:2] == (3 + group, 19 + group)
        assert sorted(ranks) == list(range(layout.hidden_size))
    incoming[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        rank_widths(net)


@pytest.mark.parametrize("family", ["swin", "split", "vit"])
def test_narrowed_layers_propagate_gradients_and_validate_master_shapes(family):
    net = _block(family)
    spec = _case(net)
    apply_spec(net, spec)
    for module in list(net.modules()):
        for name, value in list(module._buffers.items()):
            if value is not None:
                del module._buffers[name]
                module.register_parameter(name, nn.Parameter(value.clone()))
    channels = next(iter(layouts(net.plan).values())).channels
    with distillation_mode():
        _run(net, quant.q8(torch.randn(1, channels, 2, 3))).square().mean().backward()
    gradients = [parameter.grad for parameter in net.parameters()]
    assert gradients and all(grad is not None and torch.isfinite(grad).all() for grad in gradients)
    assert any(torch.count_nonzero(grad) > 0 for grad in gradients)
    validate_applied_spec(net, spec)
    with pytest.raises(ValueError, match="before promoting"):
        apply_spec(net, spec)


def test_applier_rejects_invalid_specs_without_mutating_live_candidate():
    net = _block("swin")
    applier = PruningApplier(net)
    spec = _case(net)
    applier.apply(spec)
    state = {name: value.clone() for name, value in net.state_dict().items()}
    with pytest.raises(ValueError, match="out of bounds"):
        applier.apply(PruningSpec(widths=(WidthSpec(6, heads=(2,)),)))
    assert net.pruning_spec == spec
    assert all(torch.equal(value, net.state_dict()[name]) for name, value in state.items())
    with pytest.raises(ValueError, match="full-width"):
        PruningApplier(net)
    with pytest.raises(ValueError, match="full-width"):
        rank_widths(net)
    net.layers[0].params.attention_heads = 2
    with pytest.raises(ValueError, match="structure was modified"):
        validate_applied_spec(net, spec)


def test_structure_check_does_not_materialize_parameterizations():
    net = _block("swin")
    spec = _case(net)
    apply_spec(net, spec)
    p = net.layers[0].params
    calls = []

    class Count(nn.Module):
        def forward(self, value):
            calls.append(1)
            return value

    parametrize.register_parametrization(p, "expand", Count())
    calls.clear()
    validate_applied_spec(net, spec)
    assert not calls
    p.parametrizations.expand.original = nn.Parameter(torch.zeros(1, 64))
    with pytest.raises(ValueError, match="structure was modified"):
        validate_applied_spec(net, spec)


def test_fused_kernel_rejects_reduced_hidden_shape():
    pytest.importorskip("triton")
    from dlssnr.swin_fused import supported

    net = _block("swin1")
    fake_cuda = SimpleNamespace(is_cuda=True)
    assert supported(net.layers[0].params, fake_cuda)
    apply_spec(net, _case(net, heads=False))
    assert not supported(net.layers[0].params, fake_cuda)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is unavailable")
def test_gpu_narrowed_swin_runs_and_preserves_external_shape():
    net = _block("swin1").cuda()
    spec = _case(net, heads=False)
    apply_spec(net, spec)
    x = quant.q8(torch.randn(1, 32, 2, 3, device="cuda"))
    actual = _run(net, x)
    assert actual.shape == x.shape and torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is unavailable")
@pytest.mark.parametrize("kept", [(1,), (0, 1, 3)])
def test_gpu_attention_kernels_support_compact_head_layouts(kept):
    pytest.importorskip("triton")
    from dlssnr import attention as kernels

    # Each head is independent: compacting it must preserve its exact result.
    device = "cuda"
    heads, tokens = 4, 64
    index = torch.tensor(kept, device=device)
    window_qkv = torch.randn(tokens, heads, 3, 32, device=device) * 0.1
    scale = torch.ones(heads, device=device)
    bias = torch.full((heads, tokens, tokens), swin.EXP_OFFSET, device=device)
    full = kernels.window_attention(window_qkv.reshape(tokens, -1), scale, bias)
    compact = kernels.window_attention(
        window_qkv.index_select(1, index).reshape(tokens, -1),
        scale.index_select(0, index), bias.index_select(0, index))
    expected = full.reshape(tokens, heads, 32).index_select(1, index).reshape(tokens, -1)
    assert torch.equal(compact, expected)

    global_qkv = quant.q8(torch.randn(1, 3, heads, 32, 2, 3, device=device))
    padding = float(vit.f16(torch.tensor(58 * float(vit._approximate_exp(torch.zeros(1))[0]))))
    full = kernels.vit_attention(global_qkv.reshape(1, -1, 2, 3), padding)
    compact = kernels.vit_attention(global_qkv.index_select(2, index).reshape(1, -1, 2, 3), padding)
    expected = full.reshape(1, heads, 32, 2, 3).index_select(1, index).reshape(1, -1, 2, 3)
    assert torch.equal(compact, expected)
