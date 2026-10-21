"""CLI dispatch and error reporting without constructing the actual model."""
from __future__ import annotations

import re

import pytest
import torch

from scripts import distill


def test_prepare_dispatches_paths_device_and_progress(monkeypatch, capsys):
    calls = []

    def prepare(*arguments):
        calls.append(arguments)
        arguments[-1]("preparing fake teacher")
        return {"images": [{"name": "one.png"}]}

    monkeypatch.setattr(distill, "prepare_cache", prepare)
    assert distill.main(["prepare", "--input", "images", "--out", "targets",
                         "--weights", "source.safetensors", "--device", "cpu"]) == 0
    assert calls[0][:-1] == ("images", "targets", "source.safetensors", "cpu")
    output = capsys.readouterr().out
    assert "preparing fake teacher" in output and "Teacher cache complete: 1 images" in output
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\]", output)


@pytest.mark.parametrize("custom", [False, True])
def test_train_dispatches_default_and_custom_configuration(monkeypatch, custom, capsys):
    calls = []

    def train(*arguments, **options):
        calls.append(arguments)
        assert options == {"tensorboard": custom}
        return {"status": "completed", "best_epoch": 2, "export_validation": {"psnr_db": 42.0},
                "improved": True}

    monkeypatch.setattr(distill, "train_distillation", train)
    arguments = ["train", "--pruning-report", "pruning.json", "--train-cache", "train",
                 "--val-cache", "validation"]
    if custom:
        arguments += ["--weights", "teacher.safetensors", "--device", "cpu", "--out", "run",
                      "--epochs", "3", "--lr", "0.02", "--min-lr", "0.001", "--accumulation", "2",
                      "--grad-clip", "0.7", "--patience", "2", "--seed", "4", "--no-checkpoint",
                      "--resume", "run/last.pt", "--tensorboard"]
    assert distill.main(arguments) == 0
    actual = calls[0]
    config = actual[6]
    assert actual[1:4] == ("pruning.json", "train", "validation")
    assert callable(actual[-1])
    if custom:
        assert actual[0] == "teacher.safetensors" and actual[4:6] == ("run", "cpu")
        assert config == distill.TrainConfig(3, 0.02, 0.001, 2, 0.7, 2, 4, False)
        assert actual[7] == "run/last.pt"
    else:
        assert actual[0] == "weights.safetensors" and actual[4:6] == ("out/distilled", "auto")
        assert config == distill.TrainConfig() and actual[7] is None
    assert "best epoch=2" in capsys.readouterr().out


@pytest.mark.parametrize("extra,report_path,expected_images", [
    ([], "evaluation.json", "evaluation_images"),
    (["--images-out", "comparisons"], "evaluation.json", "comparisons"),
    (["--no-images"], "evaluation.json", None),
    ([], None, "evaluation_images"),
])
def test_evaluate_dispatches_and_prints_metrics(monkeypatch, capsys, extra, report_path, expected_images):
    calls = []

    def evaluate(*arguments, **options):
        calls.append(arguments)
        assert options == {"images_out": expected_images}
        return {"metrics": {"psnr_db": 40.1, "psnr_uint8_db": 40.2, "ssim": None}}

    monkeypatch.setattr(distill, "evaluate_export", evaluate)
    arguments = ["evaluate", "--checkpoint", "best.safetensors", "--cache", "val", "--device", "cpu"] + extra
    if report_path is not None:
        arguments += ["--out", report_path]
    assert distill.main(arguments) == 0
    assert calls[0][:-1] == ("weights.safetensors", "best.safetensors", "val", report_path, "cpu")
    output = capsys.readouterr().out
    assert "PSNR=40.1000dB" in output and "SSIM=None" in output


@pytest.mark.parametrize("error,exit_code,expected", [
    (ValueError("bad source fingerprint"), 1, "ValueError: bad source fingerprint"),
    (FileNotFoundError("missing weights"), 1, "FileNotFoundError: missing weights"),
    (RuntimeError("invalid training graph"), 1, "RuntimeError: invalid training graph"),
    (torch.cuda.OutOfMemoryError("too large"), 1, "input resolution was not changed"),
    (KeyboardInterrupt(), 130, "epoch-boundary checkpoints are preserved"),
])
def test_known_failures_return_exit_code_and_actionable_error(monkeypatch, capsys, error, exit_code, expected):
    def fail(*arguments):
        raise error

    monkeypatch.setattr(distill, "prepare_cache", fail)
    assert distill.main(["prepare", "--input", "images", "--out", "cache"]) == exit_code
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [[], ["prepare"], ["train"], ["evaluate"],
                                      ["evaluate", "--checkpoint", "best.safetensors", "--cache", "val",
                                       "--images-out", "images", "--no-images"],
                                      ["train", "--epochs", "not-a-number"]])
def test_missing_or_invalid_arguments_fail_before_dispatch(arguments):
    with pytest.raises(SystemExit) as caught:
        distill.main(arguments)
    assert caught.value.code == 2


def test_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit) as caught:
        distill.main(["--help"])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert all(command in output for command in ("prepare", "train", "evaluate"))
