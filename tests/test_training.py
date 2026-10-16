"""Small numerical checks for exact-forward recovery-distillation arithmetic."""
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dlssnr import int8, quant, runtime
from dlssnr.layers import split_swin as split, swin, vit
from dlssnr.training import distillation_mode, fp8_ste, fp16_ste, inference_forward, is_distilling


def test_mode_is_nested_and_restored_after_exception():
    assert not is_distilling()
    with pytest.raises(RuntimeError), distillation_mode():
        with distillation_mode():
            assert is_distilling()
        assert is_distilling()
        raise RuntimeError("stop")
    assert not is_distilling()


def test_forward_decorator_respects_grad_mode():
    @inference_forward
    def twice(x):
        return x * 2

    x = torch.tensor(1.0, requires_grad=True)
    assert not twice(x).requires_grad
    with distillation_mode():
        twice(x).backward()
        with torch.no_grad():
            assert not twice(x).requires_grad
    assert x.grad.item() == 2


def test_rounding_is_exact_with_float32_straight_through_gradients():
    x = torch.tensor([-500.0, -448.0, -0.0023, 0.0, 0.43, 448.0, 500.0], requires_grad=True)
    assert torch.equal(fp8_ste(x), quant.q8(x))
    fp8_ste(x).sum().backward()
    assert torch.equal(x.grad, torch.tensor([0, 1, 1, 1, 1, 1, 0]))
    x.grad = None
    assert torch.equal(fp16_ste(x), x.half().float())
    (fp16_ste(x) * 1e6).sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 1e6))
    x.grad = None
    with distillation_mode():
        assert torch.equal(quant.q8(x), fp8_ste(x))
        quant.q8(x).sum().backward()
    assert x.grad[4] == 1


def test_distillation_bypasses_fused_compiler_and_int8_cache(monkeypatch):
    monkeypatch.setattr(runtime, "_COMPILE", True)
    monkeypatch.setattr(torch, "compile", lambda *a, **kw: pytest.fail("compiled train arithmetic"))
    fake_gpu = SimpleNamespace(is_cuda=True)
    with distillation_mode():
        assert runtime.fuse(lambda x: "eager")(fake_gpu) == "eager"
        assert runtime.triton_kernels(fake_gpu) is None
    monkeypatch.setattr(int8, "sites", {"vit"})
    monkeypatch.setattr(int8, "_cache", {})
    a = torch.randn(2, 3, requires_grad=True)
    b = torch.randn(3, 4, requires_grad=True)
    with distillation_mode():
        assert torch.equal(int8.mm(a, b, "vit"), a @ b)
        int8.mm(a, b, "vit").sum().backward()
    assert not int8._cache
    assert a.grad.isfinite().all() and b.grad.isfinite().all()


@pytest.mark.parametrize("family", ["swin", "split", "vit"])
def test_exponent_exact_forward_and_analytic_gradient(family):
    logits = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0], requires_grad=True)
    if family == "swin":
        bias = torch.full_like(logits, swin.EXP_OFFSET, requires_grad=True)
        function = lambda x: swin._softmax_numerators(x, bias)
        slope, rate = swin.EXP_SLOPE, 32.0
    elif family == "split":
        function = split.SplitQKVAttn.exponential
        slope, rate = split.SplitQKVAttn.EXP_SLOPE, 32.0
    else:
        function = vit._approximate_exp
        slope, rate = vit._EXP_SLOPE, 16.0
    reference = function(logits).float().detach()
    with distillation_mode():
        result = function(logits)
        assert torch.equal(reference, result)
        result.sum().backward()
    expected = reference * math.log(2) * rate * slope
    expected[[0, -1]] = 0
    torch.testing.assert_close(logits.grad, expected)
    assert logits.grad[1:-1].gt(0).all()
    if family == "swin":
        torch.testing.assert_close(bias.grad, expected / slope)


@pytest.mark.parametrize("family", ["swin", "split", "vit"])
@pytest.mark.parametrize("magnitude", [0.0, 1e-5, 0.3, 100.0])
def test_normalisation_exact_with_finite_gradients(family, magnitude):
    x = torch.full((2, 32), magnitude, requires_grad=True)
    if family == "swin":
        reference = x.detach() * swin._norm_scale(x.detach()).unsqueeze(-1)
        with distillation_mode():
            result = x * swin._norm_scale(x).unsqueeze(-1)
    elif family == "split":
        reference = split.SplitQKVAttn.normalise(x.detach().half()).float()
        with distillation_mode():
            result = split.SplitQKVAttn.normalise(fp16_ste(x))
    else:
        reference = (x.detach().half() * vit.VitQKV._inverse_norm(x.detach().half())).float()
        with distillation_mode():
            rounded = fp16_ste(x)
            result = fp16_ste(rounded * vit.VitQKV._inverse_norm(rounded))
    assert torch.equal(reference, result)
    result.square().sum().backward()
    assert x.grad.isfinite().all()


def _swin_params():
    torch.manual_seed(41)
    def weight(*shape):
        return nn.Parameter(quant.q8(torch.randn(*shape) * 0.06))
    params = swin.SwinParams(
        32, 1, expand=None, mid=None, contract=None, qkv=None, out_proj=None,
        exp_bias=None, head_scale=None, mlp_scale=None, attn_scale=None)
    for name, shape in {"expand": (128, 32), "contract": (32, 128),
                        "qkv": (96, 32), "out_proj": (32, 32)}.items():
        delattr(params, name)
        params.register_parameter(name, weight(*shape))
    for name, value in {"exp_bias": torch.full((1, 64, 64), swin.EXP_OFFSET),
                        "head_scale": torch.ones(1), "mlp_scale": torch.ones(32),
                        "attn_scale": torch.ones(32)}.items():
        delattr(params, name)
        params.register_parameter(name, nn.Parameter(value))
    params.i4 = None
    return params


def test_swin_block_preserves_forward_and_trains_qkv_and_bias():
    params = _swin_params()
    x = quant.q8(torch.randn(1, 32, 8, 8) * 0.2)
    with torch.no_grad():
        reference = swin.swin_block(params, x, 0, 0)
    with distillation_mode():
        result = swin.swin_block(params, x, 0, 0)
        assert torch.equal(result, reference)
        result.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in params.parameters())
    for part in params.qkv.grad.chunk(3):
        assert part.abs().sum() > 0
    assert params.exp_bias.grad.abs().sum() > 0


def test_split_attention_preserves_forward_and_trains_qkv_and_bias():
    torch.manual_seed(42)
    layer = split.SplitQKVAttn.__new__(split.SplitQKVAttn)
    nn.Module.__init__(layer)
    layer.shift_x = layer.shift_y = 0
    layer.i4 = None
    layer.qkv = nn.Parameter(quant.q8(torch.randn(512, 1536) * 0.02))
    layer.bias = nn.Parameter(torch.randn(16, 64, 64) * 0.1)
    layer.scale = nn.Parameter(torch.ones(16))
    x = quant.q8(torch.randn(1, 512, 4, 4) * 0.2)
    reference = layer(x)
    with distillation_mode():
        result = layer(x)
        assert torch.equal(result, reference)
        result.square().mean().backward()
    assert layer.qkv.grad.isfinite().all()
    for part in layer.qkv.grad.reshape(512, 16, 3, 32).unbind(2):
        assert part.abs().sum() > 0
    assert layer.bias.grad.isfinite().all() and layer.bias.grad.abs().sum() > 0
    assert layer.scale.grad.isfinite().all()


def test_vit_qkv_attention_preserves_forward_and_trains_all_projections():
    torch.manual_seed(43)
    layer = vit.VitQKV.__new__(vit.VitQKV)
    nn.Module.__init__(layer)
    layer.i4 = None
    layer.channels = 32
    layer.matmul = vit._Matmul(quant.q8(torch.randn(96, 32) * 0.02))
    del layer.matmul.weight
    layer.matmul.weight = nn.Parameter(quant.q8(torch.randn(96, 32) * 0.02))
    layer.temperature = nn.Parameter(torch.ones(1))
    attention = vit.VitAttention(SimpleNamespace(channels=32))
    x = quant.q8(torch.randn(1, 32, 3, 3) * 0.2)
    reference_qkv = layer(x)
    reference = attention(reference_qkv)
    with distillation_mode():
        qkv = layer(x)
        result = attention(qkv)
        assert torch.equal(qkv, reference_qkv)
        assert torch.equal(result, reference)
        result.square().mean().backward()
    for part in layer.matmul.weight.grad.chunk(3):
        assert part.isfinite().all() and part.abs().sum() > 0
    assert layer.temperature.grad.isfinite().all() and layer.temperature.grad.abs().sum() > 0


def test_split_half_activation_matches_inference_with_finite_backward():
    x = torch.linspace(-8, 8, 1001, requires_grad=True)
    reference = split.SplitFfwd.activation(x).detach()
    with distillation_mode():
        result = split.SplitFfwd.activation(x)
        assert torch.equal(reference, result)
        result.sum().backward()
    assert x.grad.isfinite().all() and x.grad.abs().sum() > 0


def test_half_pool_and_gated_blend_are_exact_and_differentiable():
    torch.manual_seed(44)
    x = torch.randn(1, 3, 8, 8, requires_grad=True)
    gain = torch.randn(1, 3, 1, 1, requires_grad=True)
    other = torch.randn_like(x, requires_grad=True)
    reference_pool = swin.pool2x2(x).detach()
    reference_blend = swin._gated_sum(x, gain, other).detach()
    with distillation_mode():
        pool = swin.pool2x2(x)
        blend = swin._gated_sum(x, gain, other)
        assert torch.equal(pool, reference_pool)
        assert torch.equal(blend, reference_blend)
        (pool.square().sum() + blend.square().sum()).backward()
    for item in (x, gain, other):
        assert item.grad.isfinite().all() and item.grad.abs().sum() > 0


def test_split_projection_pool_is_exact_and_differentiable():
    torch.manual_seed(45)
    layer = split.SplitProjPool.__new__(split.SplitProjPool)
    nn.Module.__init__(layer)
    layer.weight = nn.Parameter(quant.q8(torch.randn(4, 4) * 0.1))
    layer.gain = nn.Parameter(torch.ones(4).half().float())
    x, skip = (quant.q8(torch.randn(1, 4, 8, 8)) for _ in range(2))
    reference = layer(x, skip)
    with distillation_mode():
        result = layer(x, skip)
        assert all(torch.equal(a, b) for a, b in zip(reference, result))
        sum(t.square().sum() for t in result).backward()
    for p in layer.parameters():
        assert p.grad.isfinite().all() and p.grad.abs().sum() > 0
