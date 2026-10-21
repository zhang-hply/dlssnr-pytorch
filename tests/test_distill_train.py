"""CPU recovery-training lifecycle tests with real caches and tiny scalar models."""
from __future__ import annotations

from dataclasses import asdict, replace
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from dlssnr import distill_train as engine
from dlssnr.ablation import units
from dlssnr.distill_data import file_sha256
from dlssnr.io import Controls
from dlssnr.plan import build_plan


@pytest.fixture
def training(tmp_path, monkeypatch):
    source = tmp_path / "weights.safetensors"
    source.write_bytes(b"tiny source extraction weights")

    def cache(name, values, width=4, height=3):
        root = tmp_path / name
        (root / "targets").mkdir(parents=True)
        entries = []
        for index, value in enumerate(values):
            path = tmp_path / f"{name}_images" / f"{index}.png"
            path.parent.mkdir(exist_ok=True)
            frame = np.full((height, width, 3), value, dtype=np.uint8)
            Image.fromarray(frame).save(path)
            target = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255 * 0.8
            relative = f"targets/{index:06d}.safetensors"
            save_file({"target": target.contiguous()}, str(root / relative))
            entries.append({"name": path.name, "input_path": str(path), "input_sha256": file_sha256(path),
                            "target": relative, "ready": True, "target_sha256": file_sha256(root / relative)})
        manifest = {"schema_version": 1, "complete": True, "width": width, "height": height,
                    "weights_sha256": file_sha256(source), "weights_path": str(source),
                    "controls": asdict(Controls()), "images": entries}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return root

    class FakeNetwork(nn.Module):
        def __init__(self, width, height):
            super().__init__()
            self.width, self.height = width, height
            self.plan = build_plan(width, height)
            self.skip = set()
            self.register_buffer("gain", torch.tensor(0.5))

        def run(self, tensor, controls, capture=False):
            assert controls == Controls() and not capture
            return {self.plan[-1].index: tensor * self.gain}

    students = []

    class FakeStudent(nn.Module):
        interrupt_forward = None

        def __init__(self, net, dropped_blocks, checkpoint_blocks=True):
            super().__init__()
            self.width, self.height = net.width, net.height
            self.gain = nn.Parameter(net.gain.detach().clone())
            self.forward_calls = 0
            students.append(self)

        def forward(self, tensor):
            self.forward_calls += 1
            if self.forward_calls == type(self).interrupt_forward:
                raise KeyboardInterrupt("simulated stop")
            return tensor[..., :self.height, :self.width] * self.gain

        def export_state(self):
            return {"gain": self.gain.detach().cpu().clone()}

    monkeypatch.setattr(engine, "_new_network", lambda path, width, height: FakeNetwork(width, height))
    monkeypatch.setattr(engine, "DistilledStudent", FakeStudent)
    pruning = tmp_path / "ablation.json"
    dropped = list(units(build_plan(4, 3)))
    pruning.write_text(json.dumps({"metadata": {"width": 4, "height": 3}, "best": {"blocks": dropped}}))
    return {"source": source, "train": cache("train", [40, 70, 100]), "val": cache("val", [130]),
            "out": tmp_path / "distilled", "pruning": pruning, "cache": cache, "students": students,
            "student_class": FakeStudent,
            "config": engine.TrainConfig(epochs=2, learning_rate=0.05, min_learning_rate=0.01,
                                          accumulation=2, checkpoint_blocks=False)}


def train(settings, **overrides):
    arguments = {"weights_path": settings["source"], "pruning_report": settings["pruning"],
                 "train_cache": settings["train"], "val_cache": settings["val"], "out_dir": settings["out"],
                 "device": "cpu", "config": settings["config"], "log": lambda _: None}
    arguments.update(overrides)
    return engine.train_distillation(**arguments)


def test_training_improves_baseline_exports_and_reloads(training):
    report = train(training)
    assert report["status"] == "completed" and report["stop_reason"] == "epochs_completed"
    assert report["completed_epochs"] == 2 and report["global_step"] == 4
    assert len(report["history"]) == 2 and report["best_epoch"] == 2
    assert report["export_validation"]["mse"] < report["baseline"]["mse"]
    assert report["improved"] and report["metadata"]["pruning"]["keep_ratio"] <= 0.8
    assert report["export_validation"]["ssim"] is None
    for name in ("last.pt", "best.pt", "best.safetensors", "best.json", "report.md", "report.json"):
        assert (training["out"] / name).is_file()
    last = torch.load(training["out"] / "last.pt", weights_only=True)
    best = torch.load(training["out"] / "best.pt", weights_only=True)
    assert last["completed_epochs"] == best["completed_epochs"] == 2
    assert last["report"]["global_step"] == 4
    net, metadata = engine.load_export(training["source"], training["out"] / "best.safetensors")
    assert metadata["weights_sha256"] == file_sha256(training["source"])
    assert net.skip == set(metadata["pruning"]["layer_indices"])
    assert net.gain.item() > 0.5
    assert torch.equal(load_file(str(training["out"] / "best.safetensors"))["gain"],
                       best["student_state"]["gain"])
    output = training["out"] / "evaluation.json"
    evaluation = engine.evaluate_export(training["source"], training["out"] / "best.safetensors",
                                        training["val"], output, "cpu", log=lambda _: None)
    assert evaluation["metrics"]["mse"] == report["export_validation"]["mse"]
    assert json.loads(output.read_text())["metrics"]["ssim"] is None
    assert not list(training["out"].glob("*.tmp"))


def test_epoch_boundary_resume_matches_uninterrupted_training(training):
    uninterrupted = train(training, out_dir=training["out"].parent / "uninterrupted")
    expected = torch.load(training["out"].parent / "uninterrupted" / "last.pt", weights_only=True)
    training["student_class"].interrupt_forward = 5  # Preflight + 3 epoch-1 frames, then epoch 2.
    with pytest.raises(KeyboardInterrupt):
        train(training)
    checkpoint = torch.load(training["out"] / "last.pt", weights_only=True)
    assert checkpoint["completed_epochs"] == 1 and checkpoint["report"]["global_step"] == 2
    assert json.loads((training["out"] / "report.json").read_text())["status"] == "interrupted"
    training["student_class"].interrupt_forward = None
    resumed = train(training, resume=training["out"] / "last.pt")
    actual = torch.load(training["out"] / "last.pt", weights_only=True)
    assert resumed["global_step"] == 4 and resumed["completed_epochs"] == 2
    assert torch.equal(actual["student_state"]["gain"], expected["student_state"]["gain"])
    assert actual["scheduler_state"] == expected["scheduler_state"]
    assert resumed["export_validation"]["mse"] == uninterrupted["export_validation"]["mse"]


def test_partial_accumulation_uses_actual_group_size(training, monkeypatch):
    gradients = []
    original = torch.optim.AdamW.step

    def record(optimizer, *args, **kwargs):
        gradients.append(optimizer.param_groups[0]["params"][0].grad.item())
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", record)
    train(training, config=replace(training["config"], epochs=1))
    order = torch.randperm(3, generator=torch.Generator().manual_seed(0)).tolist()
    values = [40, 70, 100]
    assert gradients[0] == pytest.approx(-sum(values[index] / 255 for index in order[:2]) / 2)
    assert gradients[1] == pytest.approx(-values[order[2]] / 255)


def test_early_stopping_keeps_baseline_export_when_no_validation_improvement(training, monkeypatch):
    monkeypatch.setattr(training["student_class"], "export_state", lambda self: {"gain": torch.tensor(0.5)})
    report = train(training, config=replace(training["config"], epochs=5, patience=1))
    assert report["stop_reason"] == "early_stopping" and report["completed_epochs"] == 1
    assert report["best_epoch"] == 0 and not report["improved"]
    assert report["export_validation"]["mse"] == report["baseline"]["mse"]
    exported = load_file(str(training["out"] / "best.safetensors"))
    assert exported["gain"].item() == 0.5


def test_preflight_interruption_keeps_pruning_baseline_and_epoch_zero_checkpoint(training):
    training["student_class"].interrupt_forward = 1
    with pytest.raises(KeyboardInterrupt):
        train(training)
    checkpoint = torch.load(training["out"] / "last.pt", weights_only=True)
    assert checkpoint["completed_epochs"] == 0 and checkpoint["report"]["global_step"] == 0
    assert (training["out"] / "best.safetensors").is_file()
    report = json.loads((training["out"] / "report.json").read_text())
    assert report["status"] == "interrupted" and report["baseline"] is not None


def test_distillation_loss_has_pixel_and_edge_terms():
    prediction = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = engine.distillation_loss(prediction, target)
    assert loss.item() == pytest.approx(0.55)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert engine.distillation_loss(torch.ones(1, 3, 1, 1), torch.zeros(1, 3, 1, 1)) == 1
    with pytest.raises(ValueError, match="matching NCHW"):
        engine.distillation_loss(prediction, target[0])


def test_identical_train_val_images_rejected_before_student(training):
    overlap = training["cache"]("overlap", [70])
    with pytest.raises(ValueError, match="identical source images"):
        train(training, val_cache=overlap)
    assert not training["students"] and not training["out"].exists()


def test_different_train_val_dimensions_rejected(training):
    different = training["cache"]("different", [140], width=5)
    with pytest.raises(ValueError, match="dimensions differ"):
        train(training, val_cache=different)
    assert not training["students"]


@pytest.mark.parametrize("change", ["weights", "train_manifest", "config", "directory"])
def test_resume_rejects_changed_fingerprints(training, change):
    train(training)
    checkpoint_path = training["out"] / "last.pt"
    options = {"resume": checkpoint_path}
    if change == "weights":
        training["source"].write_bytes(b"other weights")
    elif change == "train_manifest":
        path = training["train"] / "manifest.json"
        path.write_text(path.read_text() + "\n")
    elif change == "config":
        options["config"] = replace(training["config"], accumulation=3)
    else:
        options["out_dir"] = training["out"].parent / "wrong_directory"
    with pytest.raises(ValueError):
        train(training, **options)


def test_train_refuses_unrelated_output_and_input_cache(training):
    training["out"].mkdir()
    protected = training["out"] / "keep.txt"
    protected.write_text("keep")
    with pytest.raises(ValueError, match="not empty"):
        train(training)
    assert protected.read_text() == "keep"
    with pytest.raises(ValueError, match="separate"):
        train(training, out_dir=training["train"])


def test_export_checks_source_identity_and_protects_artifacts(training):
    train(training)
    checkpoint = training["out"] / "best.safetensors"
    with pytest.raises(ValueError, match="overwrite"):
        engine.evaluate_export(training["source"], checkpoint, training["val"],
                               out_path=checkpoint, device="cpu", log=lambda _: None)
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    metadata["weights_sha256"] = "0" * 64
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="original teacher weights"):
        engine.load_export(training["source"], checkpoint)


@pytest.mark.parametrize("override", [{"epochs": 0}, {"accumulation": 0}, {"patience": 0},
                                     {"learning_rate": float("nan")}, {"grad_clip": 0},
                                     {"min_learning_rate": 0.1}])
def test_invalid_training_configuration_rejected(training, override):
    with pytest.raises(ValueError):
        train(training, config=replace(training["config"], **override))
    assert not training["students"]


@pytest.mark.parametrize("experiment", ["int4", "int8", "fp8_skip"])
def test_deployment_evaluation_rejects_alternate_arithmetic(training, monkeypatch, experiment):
    from dlssnr import int4, int8, quant

    train(training)
    if experiment == "int4":
        monkeypatch.setattr(int4, "active", lambda: True)
    elif experiment == "int8":
        monkeypatch.setattr(int8, "sites", {"swin"})
    else:
        monkeypatch.setattr(quant, "skip_blocks", {2})
    with pytest.raises(ValueError, match="disable INT4, INT8"):
        engine.load_export(training["source"], training["out"] / "best.safetensors")


@pytest.fixture
def events():
    accumulator = pytest.importorskip("tensorboard.backend.event_processing.event_accumulator")

    def read(directory):
        return accumulator.EventAccumulator(str(directory / "tensorboard"),
                                           size_guidance={"scalars": 0, "images": 0, "tensors": 0}).Reload()

    return read


def test_tensorboard_records_metrics_groups_and_original_images(training, monkeypatch, events):
    training["val"] = training["cache"]("visual_validation", [130, 140, 150, 160])
    losses, inference_calls = [], []
    original_loss, original_network = engine.distillation_loss, engine._new_network

    def record_loss(prediction, target):
        loss = original_loss(prediction, target)
        if torch.is_grad_enabled():
            losses.append(float(loss.detach()))
        return loss

    def record_network(*args):
        net = original_network(*args)
        original_run = net.run

        def run(*args, **kwargs):
            inference_calls.append(1)
            return original_run(*args, **kwargs)

        net.run = run
        return net

    monkeypatch.setattr(engine, "distillation_loss", record_loss)
    monkeypatch.setattr(engine, "_new_network", record_network)
    report = train(training, tensorboard=True, config=replace(training["config"], grad_clip=0.01))
    result = events(training["out"])
    steps = result.Scalars("train/loss_step")
    assert [event.step for event in steps] == [1, 2, 3, 4]
    # Ignore preflight and average each real accumulation group, including single-frame tails.
    assert [event.value for event in steps] == pytest.approx(
        [sum(losses[1:3]) / 2, losses[3], sum(losses[4:6]) / 2, losses[6]])
    assert all(event.value > 0.01 for event in result.Scalars("train/grad_norm"))
    assert [event.value for event in result.Scalars("train/learning_rate")] == pytest.approx(
        [row["learning_rate"] for row in report["history"] for _ in range(2)])
    assert [event.value for event in result.Scalars("train/loss_epoch")] == pytest.approx(
        [row["loss"] for row in report["history"]])
    for key in ("loss", "mse", "psnr_db", "mse_uint8", "psnr_uint8_db"):
        actual = result.Scalars(f"val/{key}")
        assert [event.step for event in actual] == [0, 2, 4]
        assert [event.value for event in actual] == pytest.approx(
            [report["baseline"][key]] + [row["validation"][key] for row in report["history"]])
        final = result.Scalars(f"final/{key}")
        assert [event.step for event in final] == [4]
        assert final[0].value == pytest.approx(report["export_validation"][key])
    assert "val/ssim" not in result.Tags()["scalars"]
    assert result.Scalars("best/mse")[-1].value == pytest.approx(report["best"]["mse"])
    assert result.Scalars("train/epoch")[-1].value == 2
    assert len(inference_calls) == 16  # Four frames: baseline, two epochs, final export.
    assert training["students"][0].forward_calls == 7  # Preflight plus six training frames.
    assert set(result.Tags()["images"]) == {
        f"{prefix}/images/{index:03d}" for prefix in ("val", "final") for index in range(3)}
    images = result.Images("val/images/000")
    assert [event.step for event in images] == [0, 2, 4]
    with Image.open(io.BytesIO(images[0].encoded_image_string)) as image:
        pixels = np.asarray(image)
    assert pixels.shape == (3, 12, 3)
    assert np.all(pixels[:, :4] == 130)  # Input, unchanged native resolution.
    assert np.all(pixels[:, 4:8] == 104)  # Teacher = 0.8 * input.
    assert np.all(pixels[:, 8:] == 65)  # Pruning-only student = 0.5 * input.
    description = result.Tensors("val/images/000/description/text_summary")[0].tensor_proto.string_val[0]
    assert b"0.png" in description and b"input | teacher | student" in description


def test_tensorboard_resume_hides_abandoned_updates(training, events):
    reference_dir = training["out"].parent / "continuous"
    reference = train(training, out_dir=reference_dir)
    training["student_class"].interrupt_forward = 7  # Epoch 2 already wrote optimizer step 3.
    with pytest.raises(KeyboardInterrupt):
        train(training, tensorboard=True)
    assert [e.step for e in events(training["out"]).Scalars("train/loss_step")] == [1, 2, 3]
    checkpoint = torch.load(training["out"] / "last.pt", weights_only=True)
    assert checkpoint["report"]["global_step"] == 2
    training["student_class"].interrupt_forward = None
    resumed = train(training, tensorboard=True, resume=training["out"] / "last.pt")
    result = events(training["out"])
    assert [e.step for e in result.Scalars("train/loss_step")] == [1, 2, 3, 4]
    assert [e.step for e in result.Scalars("val/mse")] == [0, 2, 4]
    assert [e.step for e in result.Images("val/images/000")] == [0, 2, 4]
    expected = torch.load(reference_dir / "last.pt", weights_only=True)
    actual = torch.load(training["out"] / "last.pt", weights_only=True)
    assert torch.equal(actual["student_state"]["gain"], expected["student_state"]["gain"])
    assert resumed["export_validation"]["mse"] == reference["export_validation"]["mse"]


def test_tensorboard_can_start_when_resuming_legacy_checkpoint(training, events):
    training["student_class"].interrupt_forward = 5
    with pytest.raises(KeyboardInterrupt):
        train(training)
    checkpoint_path = training["out"] / "last.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    # Old schema-v1 reports predate validation loss and TensorBoard.
    for metrics in [checkpoint["report"]["baseline"], checkpoint["report"]["best"]] + [
            row["validation"] for row in checkpoint["report"]["history"]]:
        metrics.pop("loss", None)
    torch.save(checkpoint, checkpoint_path)
    training["student_class"].interrupt_forward = None
    train(training, tensorboard=True, resume=checkpoint_path)
    result = events(training["out"])
    assert [e.step for e in result.Scalars("train/loss_step")] == [3, 4]
    assert [e.step for e in result.Scalars("val/loss")] == [4]
    assert [e.step for e in result.Images("val/images/000")] == [4]


@pytest.mark.parametrize("failure", [None, KeyboardInterrupt("stop"), RuntimeError("failed forward")])
def test_tensorboard_closes_on_success_and_preflight_failure(training, monkeypatch, events, failure):
    writer_type = engine.load_summary_writer()
    writers = []

    class TrackingWriter(writer_type):
        closed = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            writers.append(self)

        def close(self):
            super().close()
            self.closed = True

    monkeypatch.setattr(engine, "load_summary_writer", lambda: TrackingWriter)
    if failure is None:
        train(training, tensorboard=True)
    else:
        def fail(*args):
            raise failure

        monkeypatch.setattr(training["student_class"], "forward", fail)
        with pytest.raises(type(failure), match=str(failure)):
            train(training, tensorboard=True)
    assert len(writers) == 1 and writers[0].closed
    result = events(training["out"])
    assert result.Scalars("val/mse")[0].step == 0
    if failure is not None:
        assert "train/loss_step" not in result.Tags()["scalars"]


def test_tensorboard_missing_dependency_only_fails_when_enabled(training, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def without_tensorboard(name, *args, **kwargs):
        if name == "torch.utils.tensorboard":
            raise ImportError("simulated missing tensorboard")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_tensorboard)
    with pytest.raises(RuntimeError, match="uv pip install tensorboard"):
        train(training, tensorboard=True)
    assert not training["students"] and not training["out"].exists()
    train(training)
    assert not (training["out"] / "tensorboard").exists()


def test_tensorboard_infinite_psnr_is_text_and_ssim_none_is_omitted(training, monkeypatch, events):
    monkeypatch.setattr(training["student_class"], "export_state", lambda self: {"gain": torch.tensor(0.8)})
    report = train(training, tensorboard=True)
    result = events(training["out"])
    assert report["baseline"]["psnr_db"] == float("inf")
    assert result.Scalars("val/mse")[0].value == 0
    assert "val/psnr_db" not in result.Tags()["scalars"]
    assert "val/ssim" not in result.Tags()["scalars"]
    text = result.Tensors("val/psnr_db/nonfinite/text_summary")[0].tensor_proto.string_val[0]
    assert text == b"inf"
