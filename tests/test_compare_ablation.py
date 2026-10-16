"""Replay a reported pruning mask without expensive weights or a new search."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from dlssnr import compare_ablation as comparison
from dlssnr.ablation import layer_macs, units
from dlssnr.io import Controls
from dlssnr.plan import build_plan


def _image(path, value=20, shape=(5, 7, 3)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(shape, value, dtype=np.uint8)).save(path)
    return path


def _report(tmp_path, inputs, blocks=(3,), controls=None, width=7, height=5):
    path = tmp_path / "ablation.json"
    path.write_text(json.dumps({
        "status": "completed",
        "metadata": {
            "inputs": [{"input": str(path.resolve()), "reference": None} for path in inputs],
            "width": width, "height": height,
            "controls": asdict(controls or Controls()),
        },
        # Deliberately stale fields: only block IDs should determine the mask/cost.
        "best": {"blocks": list(blocks), "layer_indices": [999999],
                 "macs": 1, "keep_ratio": 0.01},
    }), encoding="utf-8")
    return path


def _mock_network(monkeypatch, fail_at=None, error_type=RuntimeError):
    instances = []

    class FakeNet:
        def __init__(self, weights, width, height):
            self.plan = build_plan(width, height)
            self.skip = set()
            self.calls = []
            instances.append(self)

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, frame, controls):
            self.calls.append({"skip": set(self.skip), "controls": asdict(controls),
                               "input": frame.copy()})
            if len(self.calls) == fail_at:
                raise error_type("simulated inference failure")
            delta = 4 if self.skip and frame[0, 0, 0] else 0
            return (frame.astype(np.int16) + delta).astype(np.uint8)

    monkeypatch.setattr(comparison, "NRNet", FakeNet)
    monkeypatch.setattr(comparison.Weights, "load", lambda path: object())
    return instances


def _strict_json(path):
    def reject(value):
        raise AssertionError(f"Non-standard JSON numeric literal: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _run(report, out, **kwargs):
    weights = Path(report).parent / "weights.safetensors"
    weights.write_bytes(b"mocked weights")
    return comparison.compare_ablation(report_path=report, out_dir=out, device="cpu",
                                       weights_path=weights, log=lambda message: None, **kwargs)


def test_replay_saves_real_baseline_pruned_and_ordered_side_by_side(tmp_path, monkeypatch):
    instances = _mock_network(monkeypatch)
    source = _image(tmp_path / "source.png")
    original = source.read_bytes()
    controls = Controls(intensity=0.7, style=2, auto_mask=False, seed=17)
    report = _report(tmp_path, [source], controls=controls)
    report_contents = report.read_bytes()
    out = tmp_path / "comparison"

    manifest = _run(report, out)

    assert manifest["status"] == "completed"
    assert manifest["controls"] == asdict(controls)
    assert manifest["dropped_blocks"] == [3]
    plan = build_plan(7, 5)
    expected_indices = list(units(plan)[3].indices)
    assert manifest["layer_indices"] == expected_indices
    total = sum(layer_macs(plan).values())
    assert manifest["total_macs"] == total
    assert manifest["macs"] == total - units(plan)[3].macs
    assert manifest["keep_ratio"] == pytest.approx(manifest["macs"] / total)
    assert manifest["keep_ratio"] > 0.8, "Comparison must replay masks independently of a new budget."

    assert len(manifest["images"]) == 1
    row = manifest["images"][0]
    for key in ("baseline", "pruned", "side_by_side"):
        assert row[key] == f"{key}/0001_source.png"
    baseline = np.asarray(Image.open(out / row["baseline"]))
    pruned = np.asarray(Image.open(out / row["pruned"]))
    together = np.asarray(Image.open(out / row["side_by_side"]))
    assert np.array_equal(baseline, np.full((5, 7, 3), 20, dtype=np.uint8))
    assert np.array_equal(pruned, np.full((5, 7, 3), 24, dtype=np.uint8))
    assert together.shape == (5, 14, 3)
    assert np.array_equal(together[:, :7], baseline)
    assert np.array_equal(together[:, 7:], pruned)
    assert row["mse"] == 16
    assert row["psnr_db"] == pytest.approx(10 * math.log10(255 ** 2 / 16))
    assert row["ssim"] is None
    assert source.read_bytes() == original
    assert report.read_bytes() == report_contents
    assert len(instances) == 1
    assert [call["skip"] for call in instances[0].calls] == [set(), set(expected_indices)]
    assert all(call["controls"] == asdict(controls) for call in instances[0].calls)
    assert instances[0].skip == set()
    assert _strict_json(out / "comparison.json")["status"] == "completed"


def test_directory_override_orders_inputs_and_avoids_duplicate_stem_collisions(tmp_path, monkeypatch):
    instances = _mock_network(monkeypatch)
    frames = tmp_path / "frames"
    second = _image(frames / "same.png", value=20)
    first = _image(frames / "same.bmp", value=0)
    _image(frames / "nested" / "ignored.png")
    (frames / "notes.txt").write_text("ignored")
    # Explicit input override must work even when reported originals are missing.
    report = _report(tmp_path, [tmp_path / "unavailable.png"])
    out = tmp_path / "comparison"

    manifest = _run(report, out, input_path=frames)

    assert [Path(row["input"]) for row in manifest["images"]] == [first.resolve(), second.resolve()]
    assert [Path(row["baseline"]).name for row in manifest["images"]] == [
        "0001_same.png", "0002_same.png"]
    assert len(list((out / "baseline").glob("*.png"))) == 2
    # Aggregate pixel errors, rather than averaging per-image PSNR (first is infinite).
    assert manifest["aggregate"]["mse"] == 8.0
    assert manifest["aggregate"]["psnr_db"] == pytest.approx(10 * math.log10(255 ** 2 / 8))
    assert len(instances[0].calls) == 4
    saved = _strict_json(out / "comparison.json")
    assert saved["images"][0]["psnr_db"] == "inf"


def test_markdown_report_resolves_sibling_json(tmp_path, monkeypatch):
    instances = _mock_network(monkeypatch)
    report = _report(tmp_path, [_image(tmp_path / "frame.png")])
    markdown = report.with_suffix(".md")
    markdown.write_text("# Human-readable report\n", encoding="utf-8")
    manifest = _run(markdown, tmp_path / "comparison")
    assert manifest["status"] == "completed"
    assert len(instances[0].calls) == 2


def test_markdown_without_json_fails_before_network_load(tmp_path, monkeypatch):
    instances = _mock_network(monkeypatch)
    markdown = tmp_path / "ablation.md"
    markdown.write_text("# Report without machine-readable sibling\n", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        _run(markdown, tmp_path / "comparison")
    assert not instances


def test_ssim_compares_pruned_output_to_full_baseline(tmp_path, monkeypatch):
    _mock_network(monkeypatch)
    source = _image(tmp_path / "frame.png", shape=(7, 9, 3))
    report = _report(tmp_path, [source], width=9, height=7)
    calls = []

    def fake_ssim(output, baseline):
        calls.append((output.copy(), baseline.copy()))
        return 0.875

    monkeypatch.setattr(comparison, "ssim", fake_ssim)
    manifest = _run(report, tmp_path / "comparison")
    assert manifest["images"][0]["ssim"] == 0.875
    assert manifest["aggregate"]["ssim"] == 0.875
    assert len(calls) == 1
    assert np.all(calls[0][0] == 24)
    assert np.all(calls[0][1] == 20)


@pytest.mark.parametrize("blocks", [[0], [999], [3, 3], [True], [3.0]])
def test_invalid_or_unsafe_blocks_fail_before_network_load(tmp_path, monkeypatch, blocks):
    instances = _mock_network(monkeypatch)
    report = _report(tmp_path, [_image(tmp_path / "frame.png")], blocks=blocks)
    with pytest.raises(ValueError):
        _run(report, tmp_path / "comparison")
    assert not instances


@pytest.mark.parametrize("change", ["no_best", "mixed_sizes", "reported_size", "invalid_controls"])
def test_invalid_inputs_fail_before_network_load(tmp_path, monkeypatch, change):
    instances = _mock_network(monkeypatch)
    source = _image(tmp_path / "first.png")
    report = _report(tmp_path, [source])
    data = json.loads(report.read_text())
    if change == "no_best":
        data["best"] = None
    elif change == "mixed_sizes":
        other = _image(tmp_path / "second.png", shape=(6, 7, 3))
        data["metadata"]["inputs"].append({"input": str(other), "reference": None})
    elif change == "reported_size":
        data["metadata"]["width"] = 8
    else:
        data["metadata"]["controls"]["intensity"] = 3
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        _run(report, tmp_path / "comparison")
    assert not instances


def test_nonempty_output_is_not_overwritten(tmp_path, monkeypatch):
    instances = _mock_network(monkeypatch)
    source = _image(tmp_path / "frame.png")
    report = _report(tmp_path, [source])
    out = tmp_path / "comparison"
    out.mkdir()
    sentinel = out / "existing.txt"
    sentinel.write_text("keep this")
    with pytest.raises((ValueError, FileExistsError)):
        _run(report, out)
    assert sentinel.read_text() == "keep this"
    assert not instances


@pytest.mark.parametrize("error_type,status", [(KeyboardInterrupt, "interrupted"), (RuntimeError, "error")])
def test_partial_progress_and_skip_cleanup_when_second_image_fails(tmp_path, monkeypatch, error_type, status):
    instances = _mock_network(monkeypatch, fail_at=4, error_type=error_type)
    report = _report(tmp_path, [_image(tmp_path / "first.png"), _image(tmp_path / "second.png")])
    out = tmp_path / "comparison"
    try:
        manifest = _run(report, out)
    except error_type:
        manifest = _strict_json(out / "comparison.json")
    assert manifest["status"] == status
    assert len(manifest["images"]) == 1
    assert all((out / manifest["images"][0][key]).is_file()
               for key in ("baseline", "pruned", "side_by_side"))
    assert instances[0].skip == set()
    assert _strict_json(out / "comparison.json")["status"] == status
