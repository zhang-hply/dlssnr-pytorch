"""Real compact-matrix training, inference export and structured report inputs."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import io
import json

import numpy as np
from PIL import Image
import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from dlssnr import distill_train as engine
from dlssnr import distillation
from dlssnr import compare_ablation as comparison
from dlssnr import io as frame_io
from dlssnr.distill_data import file_sha256
from dlssnr.distillation import DistilledStudent, read_pruning_report, validate_pruning
from dlssnr.io import Controls
from dlssnr.layers import vit
from dlssnr.plan import build_plan
from dlssnr.pruning import PruningSpec, WidthSpec, layouts, validate_spec
from dlssnr.width_pruning import apply_spec
from test_distillation_student import _mask, _net_from_layers, _SmallWeights, _small_swin_net


class _InputFeatures(nn.Module):
    """Buffer-free adapter; all learned arithmetic is the real ViT block."""

    def forward(self, image):
        return image.repeat(1, 22, 1, 1)[:, :64].contiguous(), image


class _MainFeatures(nn.Module):
    def forward(self, previous):
        return previous[0]


class _OutputFeatures(nn.Module):
    def forward(self, features, skip, frame):
        return features[:, :3] + skip * 0.1


def _small_vit_net():
    """A complete 64-channel, two-head ViT with NRNet's real residual routing."""
    source = {(layer.block, layer.layer): layer for layer in build_plan(320, 320)}
    extent = {"w": 8, "h": 8, "out_w": 8, "out_h": 8, "grid_x": 1, "grid_y": 1}
    plan = [replace(source[0, 0], index=0, **extent),
            # The first ViT residual reads block30.layer4 in the real graph.
            # A buffer-free projection exposes the adapter's main tensor here.
            replace(source[24, 1], index=1, block=30, layer=4, channels=64,
                    c_in=64, c_out=64, **extent)]
    for layer in range(5):
        plan.append(replace(source[31, layer], index=layer + 2, channels=64,
                            c_in=256 if layer == 1 else 64,
                            c_out=256 if layer == 0 else 64, **extent))
    plan.append(replace(source[70, 0], index=7, **extent))
    weights = _SmallWeights()
    layers = [_InputFeatures(), _MainFeatures()]
    layers.extend(vit.BUILDERS[layer.kind](layer, weights) for layer in plan[2:7])
    layers.append(_OutputFeatures())
    return _net_from_layers(plan, layers)


def _case(family):
    if family == "swin":
        # Retain non-prefix indices to exercise original-index slicing.
        spec = PruningSpec((1,), (WidthSpec(2, (tuple(range(32, 96)),)),))
        return _small_swin_net, spec, "layers.2.params.expand", (64, 32)
    spec = PruningSpec(widths=(WidthSpec(31, (tuple(range(64, 192)),), (1,)),))
    return _small_vit_net, spec, "layers.2.matmul.weight", (128, 64)


def _image(seed=37):
    return torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(seed))


def _infer(net, image):
    with torch.no_grad():
        result = net.run(image, Controls())[net.plan[-1].index]
    return result[..., :net.height, :net.width]


@pytest.mark.parametrize("family", ["swin", "vit"])
def test_compact_student_matches_physical_inference_and_exports_narrow_shapes(family):
    builder, spec, key, shape = _case(family)
    original = builder()
    original_keys = original.state_dict().keys()
    inference = apply_spec(copy.deepcopy(original), spec)
    student = DistilledStudent(original, pruning_spec=spec, checkpoint_blocks=False)
    image = _image()
    student.eval()
    assert torch.equal(student(image), _infer(inference, image))
    assert student.pruning["spec"] == spec.to_dict()
    assert student.pruning["keep_ratio"] <= 0.8
    assert student.net.pruning_spec == spec
    exported = student.export_state()
    assert exported.keys() == original_keys
    assert exported[key].shape == shape
    assert all(value.device.type == "cpu" for value in exported.values())
    assert all(parameter.dtype == torch.float32 for parameter in student.parameters())
    assert all("parametrizations" not in key for key in exported)
    inference.load_state_dict(exported, strict=True)
    assert torch.equal(student(image), _infer(inference, image))
    if family == "swin":
        assert not any(name.startswith("net.layers.1.") for name, _ in student.named_parameters())
    else:
        assert exported["layers.4.matmul.weight"].shape == (96, 64)
        assert exported["layers.6.matmul.weight"].shape == (64, 32)
        assert student.net.layers[4].channels == student.net.layers[5].channels == 32


@pytest.mark.parametrize("family", ["swin", "vit"])
def test_compact_eager_and_activation_checkpoint_gradients_match(family):
    builder, spec, _, _ = _case(family)
    eager = DistilledStudent(builder(), pruning_spec=spec, checkpoint_blocks=False)
    checked = DistilledStudent(builder(), pruning_spec=spec, checkpoint_blocks=True)
    image = _image()
    expected, actual = eager(image), checked(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    names = dict(eager.named_parameters())
    assert names.keys() == dict(checked.named_parameters()).keys()
    for name, parameter in checked.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, names[name].grad, rtol=0, atol=0)
    matrices = [parameter.grad for name, parameter in checked.named_parameters()
                if any(key in name for key in ("expand.original", "matmul.parametrizations.weight"))]
    assert matrices and any(torch.count_nonzero(gradient) for gradient in matrices)


@pytest.mark.parametrize("family", ["swin", "vit"])
def test_compact_master_and_optimizer_checkpoint_resume_matches_next_update(family):
    builder, spec, _, _ = _case(family)
    student = DistilledStudent(builder(), pruning_spec=spec, checkpoint_blocks=False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.001, weight_decay=0)

    def step(model, opt, image):
        opt.zero_grad(set_to_none=True)
        loss = model(image).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss.detach()

    before = {name: value.clone() for name, value in student.named_parameters()}
    step(student, optimizer, _image())
    assert any(not torch.equal(value, before[name]) for name, value in student.named_parameters())
    buffer = io.BytesIO()
    torch.save({"student": student.state_dict(), "optimizer": optimizer.state_dict(),
                "pruning_spec": spec.to_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    restored_spec = PruningSpec.from_dict(checkpoint["pruning_spec"])
    resumed = DistilledStudent(builder(), pruning_spec=restored_spec, checkpoint_blocks=False)
    resumed.load_state_dict(checkpoint["student"], strict=True)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.001, weight_decay=0)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    assert step(student, optimizer, _image(38)) == step(resumed, resumed_optimizer, _image(38))
    for name, parameter in resumed.state_dict().items():
        assert torch.equal(parameter, student.state_dict()[name]), name


@pytest.mark.parametrize("family", ["swin", "vit"])
def test_load_export_reconstructs_compact_shapes_before_strict_state_loading(family, tmp_path, monkeypatch):
    builder, spec, key, shape = _case(family)
    student = DistilledStudent(builder(), pruning_spec=spec, checkpoint_blocks=False)
    source = tmp_path / "teacher.safetensors"
    source.write_bytes(b"original teacher extraction identity")
    destination = tmp_path / "compact.safetensors"
    metadata = {"schema_version": 2, "weights_sha256": file_sha256(source),
                "width": student.net.width, "height": student.net.height,
                "controls": asdict(Controls()), "dropped_blocks": list(spec.dropped_blocks),
                "pruning_spec": spec.to_dict(), "pruning": student.pruning}
    original_plan = builder().plan
    monkeypatch.setattr(engine, "build_plan", lambda width, height: original_plan)
    monkeypatch.setattr(engine, "_new_network", lambda path, width, height: builder())
    engine._export(student, metadata, destination)
    assert load_file(str(destination))[key].shape == shape
    restored, actual_metadata = engine.load_export(source, destination, "cpu")
    assert actual_metadata == metadata
    assert restored.pruning_spec == spec
    assert restored.state_dict()[key].shape == shape
    assert torch.equal(_infer(restored, _image()), student.eval()(_image()))

    # Every required tensor must still be present; compact loading is strict.
    state = load_file(str(destination))
    del state[key]
    save_file(state, str(destination))
    with pytest.raises(RuntimeError, match="Missing key"):
        engine.load_export(source, destination, "cpu")


@pytest.mark.parametrize("mutation", ["head_count", "hidden_shape", "mask", "spec"])
def test_forward_rejects_changes_to_the_fixed_structured_student(mutation):
    builder, spec, _, _ = _case("vit")
    student = DistilledStudent(builder(), pruning_spec=spec, checkpoint_blocks=False)
    if mutation == "head_count":
        student.net.layers[4].channels = 64
    elif mutation == "hidden_shape":
        original = student.net.layers[2].matmul.parametrizations.weight.original
        original.data = original.data[:64].clone()
    elif mutation == "mask":
        student.net.skip = {2, 3, 4, 5, 6}
    else:
        student.net.pruning_spec = PruningSpec()
    with pytest.raises(ValueError, match="modified"):
        student(_image())


def _report_cases():
    plan = build_plan(320, 320)
    depth = PruningSpec(tuple(_mask(plan)))
    width = PruningSpec(widths=tuple(WidthSpec(
        block, tuple(tuple(range(layout.hidden_size // 2)) for _ in range(layout.groups)),
        tuple(range(max(1, layout.heads // 2)))) for block, layout in layouts(plan).items()))
    hybrid = PruningSpec(depth.dropped_blocks, tuple(
        entry for entry in width.widths if entry.block not in depth.dropped_blocks))
    return plan, {"depth": depth, "width": width, "hybrid": hybrid}


def _report(specs):
    def row(spec):
        return {"pruning_spec": spec.to_dict(), "macs": 0, "layer_indices": [999]}
    return {"schema_version": 2,
            "metadata": {"width": 320, "height": 320, "weights_sha256": "teacher-hash",
                         "controls": asdict(Controls())},
            "best": row(specs["hybrid"]),
            "methods": {name: {"best": row(spec)} for name, spec in specs.items()}}


@pytest.mark.parametrize("method", ["best", "depth", "width", "hybrid"])
def test_v2_report_selects_requested_method_and_recomputes_untrusted_stats(method, tmp_path):
    plan, specs = _report_cases()
    path = tmp_path / "pruning.json"
    path.write_text(json.dumps(_report(specs)))
    result = read_pruning_report(path, plan, plan_method=method,
                                 weights_hash="teacher-hash", controls=asdict(Controls()))
    expected = specs["hybrid" if method == "best" else method]
    stats = validate_spec(plan, expected, keep_ratio=0.8)
    assert all(result[key] == value for key, value in stats.items())
    assert result["plan_method"] == method
    assert result["macs"] > 0 and 999 not in result["layer_indices"]


@pytest.mark.parametrize("mismatch", ["weights", "controls", "dimensions", "missing_method", "missing_spec", "wrong_kind"])
def test_v2_report_rejects_wrong_identity_or_unavailable_structure(mismatch, tmp_path):
    plan, specs = _report_cases()
    report = _report(specs)
    if mismatch == "weights":
        report["metadata"]["weights_sha256"] = "other-teacher"
    elif mismatch == "controls":
        report["metadata"]["controls"]["style"] = (Controls().style + 1) % 3
    elif mismatch == "dimensions":
        report["metadata"]["width"] = 640
    elif mismatch == "missing_method":
        del report["methods"]["width"]
    elif mismatch == "missing_spec":
        del report["methods"]["width"]["best"]["pruning_spec"]
    else:
        report["methods"]["width"]["best"]["pruning_spec"] = specs["depth"].to_dict()
    path = tmp_path / "pruning.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        read_pruning_report(path, plan, plan_method="width",
                            weights_hash="teacher-hash", controls=asdict(Controls()))


def test_legacy_report_and_student_stats_stay_compatible(tmp_path):
    net = _small_swin_net()
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"best": {"blocks": [1], "macs": 0}}))
    expected = validate_pruning(net.plan, [1])
    assert read_pruning_report(path, net.plan) == expected
    legacy = DistilledStudent(net, [1], checkpoint_blocks=False)
    assert legacy.pruning == expected
    assert legacy(_image()).shape == (1, 3, 5, 7)


def test_structured_training_lifecycle_resume_and_changed_head_rejection(tmp_path, monkeypatch):
    """Exercise the real train/export/evaluate loop with compact trainable matrices."""
    builder, spec, key, expected_shape = _case("vit")
    original_plan = builder().plan
    source = tmp_path / "weights.safetensors"
    source.write_bytes(b"tiny source identity")
    digest = file_sha256(source)
    monkeypatch.setattr(engine, "build_plan", lambda w, h: original_plan)
    monkeypatch.setattr(distillation, "build_plan", lambda w, h: original_plan)
    monkeypatch.setattr(engine, "_new_network", lambda path, w, h: builder())

    def frame_to_tensor(frame, device):
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255
        return torch.nn.functional.pad(tensor, (0, 1, 0, 3))

    monkeypatch.setattr(engine, "frame_to_tensor", frame_to_tensor)

    def cache(name, values):
        root = tmp_path / name
        (root / "targets").mkdir(parents=True)
        entries = []
        for index, value in enumerate(values):
            frame = np.full((5, 7, 3), value, dtype=np.uint8)
            image = tmp_path / f"{name}_{index}.png"
            Image.fromarray(frame).save(image)
            relative = f"targets/{index}.safetensors"
            target = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255 * 0.8
            save_file({"target": target.contiguous()}, str(root / relative))
            entries.append({"name": image.name, "input_path": str(image), "input_sha256": file_sha256(image),
                            "target": relative, "target_sha256": file_sha256(root / relative), "ready": True})
        (root / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "complete": True, "width": 7, "height": 5,
            "weights_sha256": digest, "weights_path": str(source),
            "controls": asdict(Controls()), "images": entries}))
        return root

    training, validation = cache("train", [40, 70]), cache("val", [130])
    report = {"schema_version": 2, "metadata": {"width": 7, "height": 5, "weights_sha256": digest,
                                               "controls": asdict(Controls())},
              "best": {"pruning_spec": spec.to_dict()},
              "methods": {"width": {"best": {"pruning_spec": spec.to_dict()}}}}
    pruning_path = tmp_path / "pruning.json"
    pruning_path.write_text(json.dumps(report))
    config = engine.TrainConfig(epochs=2, accumulation=1, learning_rate=1e-3,
                                min_learning_rate=1e-4, checkpoint_blocks=True)

    def run(out, **kwargs):
        return engine.train_distillation(source, pruning_path, training, validation, out,
                                         device="cpu", config=config, plan_method="width", **kwargs)

    continuous = run(tmp_path / "continuous", log=lambda _: None)

    def interrupt(message):
        if message.startswith("Epoch 2/2 image 1/"):
            raise KeyboardInterrupt("after an uncommitted compact-model update")

    out = tmp_path / "resumed"
    with pytest.raises(KeyboardInterrupt):
        run(out, log=interrupt)
    saved = torch.load(out / "last.pt", weights_only=True)
    assert saved["schema_version"] == 2 and saved["completed_epochs"] == 1
    assert saved["metadata"]["pruning_spec"] == spec.to_dict()
    resumed = run(out, resume=out / "last.pt", log=lambda _: None)
    expected = torch.load(tmp_path / "continuous" / "last.pt", weights_only=True)
    actual = torch.load(out / "last.pt", weights_only=True)
    for name, value in expected["student_state"].items():
        assert torch.equal(value, actual["student_state"][name]), name
    assert resumed["export_validation"]["mse"] == continuous["export_validation"]["mse"]
    assert load_file(str(out / "best.safetensors"))[key].shape == expected_shape
    assert "Selected pruning method: width" in (out / "report.md").read_text()
    evaluation = engine.evaluate_export(source, out / "best.safetensors", validation,
                                        device="cpu", log=lambda _: None, images_out=out / "images")
    assert evaluation["comparisons"]["status"] == "completed"
    assert evaluation["metrics"]["mse"] == resumed["export_validation"]["mse"]

    # Same compact dimensions but a different original head must not resume.
    changed = PruningSpec(widths=(WidthSpec(31, spec.widths[0].mlp, (0,)),))
    report["methods"]["width"]["best"]["pruning_spec"] = changed.to_dict()
    pruning_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different pruning_spec"):
        run(out, resume=out / "last.pt", log=lambda _: None)


def test_comparison_replays_selected_width_plan_and_restores_full_baseline(tmp_path, monkeypatch):
    builder, spec, _, _ = _case("vit")
    plan = builder().plan
    source = tmp_path / "weights.safetensors"
    source.write_bytes(b"teacher source")
    monkeypatch.setattr(distillation, "build_plan", lambda w, h: plan)
    monkeypatch.setattr(comparison, "build_plan", lambda w, h: plan)
    monkeypatch.setattr(comparison.Weights, "load", lambda path: object())
    nets = []

    def network(*args):
        net = builder()
        nets.append(net)
        return net

    def frame_to_tensor(frame, device):
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255
        return torch.nn.functional.pad(tensor, (0, 1, 0, 3))

    monkeypatch.setattr(comparison, "NRNet", network)
    monkeypatch.setattr(frame_io, "frame_to_tensor", frame_to_tensor)
    frames = [np.full((5, 7, 3), value, dtype=np.uint8) for value in (80, 140)]
    images = []
    for index, frame in enumerate(frames):
        path = tmp_path / f"frame{index}.png"
        Image.fromarray(frame).save(path)
        images.append(path)
    path = tmp_path / "comparison_source.json"
    # The global best deliberately differs from the explicitly selected width plan.
    path.write_text(json.dumps({"schema_version": 2,
        "metadata": {"width": 7, "height": 5, "weights_sha256": file_sha256(source),
                     "controls": asdict(Controls()), "inputs": [{"input": str(p)} for p in images]},
        "best": {"pruning_spec": PruningSpec((31,)).to_dict()},
        "methods": {"width": {"best": {"pruning_spec": spec.to_dict()}}}}))
    out = tmp_path / "comparison"
    result = comparison.compare_ablation(path, out_dir=out, weights_path=source,
                                         device="cpu", log=lambda _: None, plan_method="width")
    assert result["pruning_spec"] == spec.to_dict() and result["plan_method"] == "width"
    full, compact = builder(), apply_spec(builder(), spec)
    for frame, row in zip(frames, result["images"]):
        with Image.open(out / row["baseline"]) as image:
            assert np.array_equal(np.asarray(image), full(frame, Controls()))
        with Image.open(out / row["pruned"]) as image:
            assert np.array_equal(np.asarray(image), compact(frame, Controls()))
    assert nets[0].pruning_spec == PruningSpec()
    assert nets[0].layers[4].channels == 64
