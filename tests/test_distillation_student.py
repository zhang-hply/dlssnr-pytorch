"""Whole-block training, checkpointing, and deployment-state equivalence."""
import copy
import json
from dataclasses import replace

import pytest
import torch
from torch import nn

from dlssnr import int8, quant
from dlssnr.ablation import layer_macs, units
from dlssnr.distillation import DistilledStudent, read_pruning_report, validate_pruning
from dlssnr.io import Controls
from dlssnr.layers import pre_post, swin, vit, split_swin
from dlssnr.model import NRNet
from dlssnr.plan import Padding, build_plan
from dlssnr.training import distillation_mode


class _SmallWeights:
    """Decoded deterministic weights for tiny real Swin and ViT modules."""

    def __init__(self):
        self.generator = torch.Generator().manual_seed(17)

    def e4m3(self, _block, _layer, _name, *shape):
        return quant.q8(torch.randn(*shape, generator=self.generator) * 0.03)

    def f16(self, _block, _layer, name, *shape):
        if shape == (-1,):
            shape = (48,) if name == "residual_scale" else (40,)
        if name == "residual_scale":
            value = torch.cat([torch.zeros(8), torch.full((32,), 0.9), torch.zeros(8)])
        elif name in {"attn_residual_scale", "skip_weight", "main_gain", "skip_gain"}:
            value = torch.full(shape, 0.8)
        else:
            value = torch.randn(*shape, generator=self.generator) * 0.03
        return value.half().float()

    def f32(self, _block, _layer, _name, *shape):
        if shape == (-1,):
            shape = (4,)
        return torch.full(shape, 1.5)


def _net_from_layers(plan, layers, width=7, height=5):
    net = NRNet.__new__(NRNet)
    nn.Module.__init__(net)
    net.width, net.height = width, height
    net.padding = Padding(8, 8, 8 - width, 8 - height)
    net.plan, net.layers = plan, nn.ModuleList(layers)
    net.skip_source = net._skip_sources()
    net.last_use = net._last_uses()
    net.skip = set()
    return net


def _small_swin_net():
    source = {(spec.block, spec.layer): spec for spec in build_plan(320, 320)}
    plan = [
        replace(source[0, 0], index=0, w=8, h=8, out_w=4, out_h=4, grid_x=1, grid_y=1),
        replace(source[1, 0], index=1, w=4, h=4, out_w=4, out_h=4, grid_x=1, grid_y=1,
                shift_x=0, shift_y=0),
        replace(source[2, 0], index=2, w=4, h=4, out_w=4, out_h=4, grid_x=1, grid_y=1,
                shift_x=0, shift_y=0),
        replace(source[70, 0], index=3, w=4, h=4, out_w=8, out_h=8, grid_x=1, grid_y=1,
                shift_x=0, shift_y=0),
    ]
    weights = _SmallWeights()
    layers = [pre_post.PreBlock(plan[0], weights), swin.SwinLayer(plan[1], weights),
              swin.SwinLayer(plan[2], weights), pre_post.PostBlock(plan[3], weights)]
    return _net_from_layers(plan, layers)


def _mask(plan, ratio=0.8):
    total = sum(layer_macs(plan).values())
    blocks, removed = [], 0
    for unit in units(plan).values():
        blocks.append(unit.block)
        removed += unit.macs
        if total - removed <= total * ratio:
            return blocks
    raise AssertionError("not enough candidates")


def test_pruning_recalculates_budget_and_rejects_unsafe_masks():
    plan = build_plan(1920, 1080)
    mask = _mask(plan)
    result = validate_pruning(plan, mask)
    assert result["keep_ratio"] <= 0.8
    assert result["total_macs"] == 508853747712
    assert result["layer_indices"] == sorted(
        spec.index for spec in plan if spec.block in mask)
    for bad in ([0], [30], [39], [70], [123], [True], [1, 1]):
        with pytest.raises(ValueError):
            validate_pruning(plan, bad)
    with pytest.raises(ValueError, match="exceeding"):
        validate_pruning(plan, [])
    for ratio in (0, -1, 1.01, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            validate_pruning(plan, mask, ratio)


def test_report_load_checks_dimensions_and_recomputes_untrusted_fields(tmp_path):
    plan = build_plan(1920, 1080)
    report = {"metadata": {"width": 1920, "height": 1080},
              "best": {"blocks": _mask(plan), "macs": 0, "layer_indices": []}}
    path = tmp_path / "ablation.json"
    path.write_text(json.dumps(report))
    result = read_pruning_report(path, plan)
    assert result["macs"] > 0 and result["layer_indices"]
    with pytest.raises(ValueError, match="dimensions"):
        read_pruning_report(path, build_plan(1280, 720))
    report["best"] = None
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="no best"):
        read_pruning_report(path, plan)


def test_forward_matches_pruned_inference_and_exports_canonical_buffers():
    inference = _small_swin_net()
    student = DistilledStudent(copy.deepcopy(inference), [1], checkpoint_blocks=False)
    inference.skip = set(student.pruning["layer_indices"])
    image = torch.rand(1, 3, 8, 8)
    with torch.no_grad():
        expected = inference.run(image, Controls())[3][..., :5, :7]
    student.eval()
    actual = student(image)
    assert torch.equal(actual, expected)
    names = dict(student.named_parameters())
    assert names and not any(name.startswith("net.layers.1.") for name in names)
    assert all(value.dtype == torch.float32 for value in names.values())
    assert any("exp_bias" in name for name in names)
    state_before = {key: value.clone() for key, value in student.state_dict().items()}
    exported = student.export_state()
    assert exported.keys() == inference.state_dict().keys()
    assert all(value.device.type == "cpu" for value in exported.values())
    assert all(torch.equal(value, state_before[key]) for key, value in student.state_dict().items())
    inference.load_state_dict(exported)
    with torch.no_grad():
        assert torch.equal(inference.run(image, Controls())[3][..., :5, :7], actual)
    assert validate_pruning(inference.plan, [1]) == student.pruning


def test_checkpoint_and_eager_training_match_and_retained_weights_update():
    eager = DistilledStudent(_small_swin_net(), [1], checkpoint_blocks=False)
    checked = DistilledStudent(_small_swin_net(), [1], checkpoint_blocks=True)
    image = torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(31))
    expected, actual = eager(image), checked(image)
    assert torch.equal(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for (name, left), (other_name, right) in zip(eager.named_parameters(), checked.named_parameters()):
        assert name == other_name
        assert left.grad is not None, name
        assert torch.isfinite(left.grad).all(), name
        assert right.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    originals = {name: value.clone() for name, value in checked.named_parameters()}
    torch.optim.SGD(checked.parameters(), lr=0.1).step()
    assert any(not torch.equal(value, originals[name]) for name, value in checked.named_parameters())
    assert not any("layers.1." in name for name, _ in checked.named_parameters())


def test_master_checkpoint_resume_preserves_subgrid_updates_and_export_quantizes():
    student = DistilledStudent(_small_swin_net(), [1])
    master = student.net.layers[2].params.parametrizations.expand.original
    with torch.no_grad():
        master.add_(0.0001)
    resumed = DistilledStudent(_small_swin_net(), [1])
    resumed.load_state_dict(student.state_dict())
    assert torch.equal(resumed.net.layers[2].params.parametrizations.expand.original, master)
    exported = student.export_state()["layers.2.params.expand"]
    assert torch.equal(quant.q8(exported), exported)
    assert not torch.equal(exported, master)


class _RoutingLayer(nn.Module):
    """Cheap arithmetic with all the tuple/residual interfaces of the real graph."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def forward(self, value, skip=None, frame=None):
        kind, variant = self.spec.kind, self.spec.variant
        if kind == "PreBlock":
            return value * 0.9, value * 1.1
        if kind == "PostBlock":
            return value * 0.9 + skip * 0.1 + frame * 0.01
        if kind == "SplitFfwd":
            return value * 0.9, value
        if kind == "SplitFfwdProj":
            return value[0] + value[1] * 0.1
        if kind == "SplitFinalHead":
            return skip[1] * 0.9
        if kind == "SplitProjPool":
            projected = value * 0.8 + skip * 0.2
            return projected, projected * 0.9
        if kind == "DecInputUpsample":
            return value * 0.9 + skip * 0.1
        if variant == "ds":
            return value * 0.9, value * 1.1
        if variant == "upsample":
            return value * 0.9 + skip[1] * 0.1
        if skip is not None:
            return value * 0.9 + skip * 0.1
        return value * 0.99 + 0.001


def test_all_152_layers_route_cross_block_skips_and_dropped_tuples_like_inference():
    plan = build_plan(320, 320)
    network = _net_from_layers(plan, [_RoutingLayer(spec) for spec in plan])
    mask = [1, 2, 23, 31, 32, 36, 47, 69]
    student = DistilledStudent(copy.deepcopy(network), mask, keep_ratio=1)
    network.skip = set(student.pruning["layer_indices"])
    image = torch.rand(1, 3, 8, 8, requires_grad=True)
    expected = network.run(image)[151][..., :5, :7]
    actual = student(image)
    assert torch.equal(actual, expected)
    expected_gradient, = torch.autograd.grad(expected.sum(), image)
    actual_gradient, = torch.autograd.grad(actual.sum(), image)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_rejects_mutated_masks_experiments_unknown_buffers_and_wrong_inputs(monkeypatch):
    net = _small_swin_net()
    net.skip = {2}
    with pytest.raises(ValueError, match="partial"):
        DistilledStudent(net, [1])
    net = _small_swin_net()
    net.layers[2].register_buffer("future_learned_weight", torch.ones(1))
    with pytest.raises(ValueError, match="unclassified"):
        DistilledStudent(net, [1])
    student = DistilledStudent(_small_swin_net(), [1])
    for image in (torch.rand(1, 3, 5, 7), torch.rand(1, 3, 8, 8).half()):
        with pytest.raises(ValueError, match="padded float32"):
            student(image)
    student.net.skip = set()
    with pytest.raises(ValueError, match="mask was modified"):
        student(torch.rand(1, 3, 8, 8))
    monkeypatch.setattr(int8, "sites", {"swin"})
    with pytest.raises(ValueError, match="disable INT4, INT8"):
        DistilledStudent(_small_swin_net(), [1])


@pytest.mark.parametrize("family", ["vit", "split"])
def test_attention_families_propagate_finite_nonzero_weight_gradients(family):
    source = {(spec.block, spec.layer): spec for spec in build_plan(320, 320)}
    if family == "vit":
        weights = _SmallWeights()
        qspec = replace(source[31, 2], channels=32, c_in=32, c_out=32)
        aspec = replace(source[31, 3], channels=32)
        projection = vit.VitQKV(qspec, weights)
        attention = vit.VitAttention(aspec)
        # Exercise the same promotion on a minimal net while keeping the actual
        # QKV and attention implementations, including normalization and FP8.
        network = _net_from_layers(
            [replace(qspec, index=0), replace(aspec, index=1)], [projection, attention])
        image = quant.q8(torch.randn(1, 32, 2, 2))
    else:
        projection = split_swin.SplitQKVAttn.__new__(split_swin.SplitQKVAttn)
        nn.Module.__init__(projection)
        projection.shift_x = projection.shift_y = 0
        projection.block, projection.i4 = 23, None
        projection.register_buffer("qkv", quant.q8(torch.randn(512, 1536) * 0.03))
        projection.register_buffer("bias", torch.zeros(16, 64, 64))
        projection.register_buffer("scale", torch.ones(16))
        network = _net_from_layers([replace(source[23, 2], index=0)], [projection])
        image = quant.q8(torch.randn(1, 512, 2, 2))
    # No complete skippable block exists in these isolated graphs; a budget of
    # one lets the constructor validate the full retained layer set.
    student = DistilledStudent(network, [], keep_ratio=1, checkpoint_blocks=False)
    with distillation_mode():
        output = student.net.layers[0](image)
        if family == "vit":
            output = student.net.layers[1](output)
        output.square().mean().backward()
    gradients = [value.grad for value in student.parameters()]
    assert gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients)
