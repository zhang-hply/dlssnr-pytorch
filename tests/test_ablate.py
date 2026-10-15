"""Pruning search tests use measured, non-additive errors without loading GPU weights."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from dlssnr.ablation import BlockUnit
from dlssnr.plan import build_plan
from scripts import ablate


def _candidates(costs=(10, 10, 10, 10)):
    return {
        block: BlockUnit(block, "swin", (block * 2, block * 2 + 1), cost)
        for block, cost in enumerate(costs, 1)
    }


def test_search_measures_interactions_and_deduplicates_candidates():
    candidates = _candidates()
    search = ablate.BlockSearch(candidates, 100, ablate.SearchConfig(
        beam_width=2, expand_top_k=3, max_evals=20,
    ))
    calls, updates = [], []

    def evaluate(blocks):
        assert blocks == tuple(sorted(set(blocks)))
        assert blocks not in calls
        calls.append(blocks)
        if len(blocks) == 1:
            return {"mse": float(blocks[0])}
        return {"mse": 0.25 if blocks == (1, 3) else 100.0}

    search.run(evaluate, lambda state, row: updates.append(tuple(row["blocks"])))
    best = search.best_results(1)[0]
    assert best["blocks"] == [1, 3]
    assert best["mse"] == 0.25
    assert best["layer_indices"] == [2, 3, 6, 7]
    assert best["macs"] == 80
    assert best["keep_ratio"] == pytest.approx(0.8)
    assert best["saved_macs"] == 20
    assert best["feasible"]
    assert best["psnr_db"] == pytest.approx(10 * math.log10(255 ** 2 / 0.25))
    assert set(calls) == set(search.results)
    assert len(updates) == len(calls)
    assert len(calls) <= 20
    assert all(len(blocks) <= 2 for blocks in calls), "Feasible combinations must be terminal."
    assert search.stop_reason


def test_search_budget_preserves_a_measured_feasible_seed():
    search = ablate.BlockSearch(_candidates(), 100, ablate.SearchConfig(max_evals=5))
    calls = []

    def evaluate(blocks):
        calls.append(blocks)
        return {"mse": float(sum(blocks) ** 2)}

    search.run(evaluate)
    assert calls[:4] == [(1,), (2,), (3,), (4,)]
    assert len(calls) == 5
    assert calls[-1] == (1, 2)
    assert search.best_results(1)[0]["blocks"] == [1, 2]
    assert search.best_results(1)[0]["keep_ratio"] == 0.8
    assert search.stop_reason


def test_final_ties_prefer_more_retained_macs_then_fewer_blocks_then_ids():
    # Block 3 is feasible alone, but deleting 1+2 retains more computation for equal error.
    search = ablate.BlockSearch(_candidates((10, 10, 25, 10)), 100,
                               ablate.SearchConfig(expand_top_k=4, max_evals=20))
    search.run(lambda blocks: {"mse": 1.0})
    best = search.best_results(10)
    assert best[0]["blocks"] == [1, 2]
    assert best[0]["macs"] == 80
    assert [row["blocks"] for row in best if row["macs"] == 80] == [[1, 2], [1, 4], [2, 4]]

    search = ablate.BlockSearch(_candidates((20, 10, 10)), 100,
                               ablate.SearchConfig(expand_top_k=3, max_evals=10))
    search.run(lambda blocks: {"mse": 0.0})
    best = search.best_results(2)
    assert [row["blocks"] for row in best] == [[1], [2, 3]]
    assert math.isinf(best[0]["psnr_db"])


@pytest.mark.parametrize("overrides", [
    {"keep_ratio": 0}, {"keep_ratio": 1}, {"keep_ratio": math.nan},
    {"beam_width": 0}, {"expand_top_k": 0}, {"max_evals": 4},
])
def test_search_rejects_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        ablate.BlockSearch(_candidates(), 100, ablate.SearchConfig(**overrides))


def test_search_rejects_invalid_costs_or_unreachable_budget():
    with pytest.raises(ValueError):
        ablate.BlockSearch(_candidates((0, 20)), 100, ablate.SearchConfig())
    with pytest.raises(ValueError):
        ablate.BlockSearch(_candidates((5, 5)), 100, ablate.SearchConfig())


def _image(path: Path, value=0, shape=(3, 4, 3)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(shape, value, dtype=np.uint8)).save(path)
    return path


def test_load_inputs_sorts_pairs_without_recursing(tmp_path):
    source, reference = tmp_path / "source", tmp_path / "reference"
    for name in ("b.PNG", "a.png"):
        _image(source / name)
        _image(reference / name, value=1)
    _image(source / "nested" / "ignored.png")
    (source / "notes.txt").write_text("not an image")
    cases = ablate.load_inputs(source, reference)
    assert [case.path.name for case in cases] == ["a.png", "b.PNG"]
    assert all(case.reference_path == reference / case.path.name for case in cases)
    assert all(case.frame.shape == (3, 4, 3) and case.reference.shape == case.frame.shape for case in cases)
    assert all(case.full is None for case in cases)
    single = ablate.load_inputs(source / "a.png")
    assert len(single) == 1 and single[0].reference is None


def test_load_inputs_rejects_shape_mismatch_and_missing_reference(tmp_path):
    source, reference = tmp_path / "source", tmp_path / "reference"
    _image(source / "a.png")
    _image(source / "b.png", shape=(1, 4, 3))
    with pytest.raises(ValueError):
        ablate.load_inputs(source)
    with pytest.raises(ValueError):
        ablate.load_inputs(source / "a.png", source / "b.png")
    reference.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        ablate.load_inputs(source, reference)


def _mock_network(monkeypatch, interrupt_at=None):
    instances = []

    class FakeNet:
        def __init__(self, weights, width, height):
            self.plan = build_plan(width, height)
            self.skip = set()
            self.calls = 0
            instances.append(self)

        def to(self, device):
            return self

        def __call__(self, frame, controls):
            self.calls += 1
            if self.calls == interrupt_at:
                raise KeyboardInterrupt
            delta = 4 if self.skip and frame[0, 0, 0] else 0
            return (frame.astype(np.int16) + delta).astype(np.uint8)

    monkeypatch.setattr(ablate, "NRNet", FakeNet)
    monkeypatch.setattr(ablate.Weights, "load", lambda path: object())
    return instances


def _strict_report(path):
    def reject_constant(value):
        raise AssertionError(f"Non-standard JSON numeric literal: {value}")

    return json.loads(path.read_text(), parse_constant=reject_constant)


@pytest.mark.parametrize("with_reference", [False, True])
def test_main_writes_reports_using_global_mse_and_cleans_up(tmp_path, monkeypatch, with_reference):
    instances = _mock_network(monkeypatch)
    source = tmp_path / "frames"
    _image(source / "a.png", value=0)
    _image(source / "b.png", value=20)
    out = tmp_path / "new" / "report.md"
    args = [
        "--input", str(source), "--device", "cpu", "--out", str(out),
        "--max-evals", "60", "--top-k", "1",
    ]
    if with_reference:
        reference = tmp_path / "reference"
        _image(reference / "a.png", value=2)
        _image(reference / "b.png", value=22)
        args.extend(["--reference", str(reference)])
    result = ablate.main(args)
    assert result == 0
    assert out.is_file() and out.read_text().strip()
    report = _strict_report(out.with_suffix(".json"))
    assert report["status"] == "completed"
    assert report["stop_reason"]
    assert len(report["single_block_results"]) == 59
    assert report["progress"]["unique_evaluations"] == 60
    assert len(report["evaluations"]) == 60
    best = report["best"]
    assert best["mse"] == 8.0
    assert best["psnr_db"] == pytest.approx(10 * math.log10(255 ** 2 / 8))
    assert best["keep_ratio"] <= 0.8
    assert best["ssim"] is None
    if with_reference:
        assert best["reference_mse"] == 4.0
    assert report["top_results"][0]["blocks"] == best["blocks"]
    calls = report["progress"]["forward_calls"]
    assert calls == {"baseline": 2, "search": 120, "ssim": 0}
    assert instances[0].calls == sum(calls.values())
    assert instances[0].skip == set()


def test_ssim_supplement_replays_top_combination_and_counts_it(tmp_path, monkeypatch):
    instances = _mock_network(monkeypatch)
    frame = np.full((12, 14, 3), 20, dtype=np.uint8)
    reference = np.full_like(frame, 22)
    case = ablate.ImageCase(tmp_path / "frame.png", frame, tmp_path / "ref.png", reference)
    net = ablate.NRNet(object(), 14, 12)
    evaluator = ablate.ModelEvaluator(net, [case], ablate.Controls(), {})
    evaluator.baseline()
    comparisons = []

    def ssim(output, target):
        comparisons.append((output.copy(), target.copy()))
        return 0.75 if target[0, 0, 0] == 20 else 0.5

    monkeypatch.setattr(ablate, "ssim", ssim)
    row = {"layer_indices": [2, 3], "per_image": [{}]}
    evaluator.supplement_ssim(row)
    assert row["ssim"] == 0.75
    assert row["reference_ssim"] == 0.5
    assert row["per_image"][0] == {"ssim": 0.75, "reference_ssim": 0.5}
    assert len(comparisons) == 2
    assert all(np.all(output == 24) for output, _ in comparisons)
    assert evaluator.forward_calls == {"baseline": 1, "search": 0, "ssim": 1}
    assert instances[0].calls == 2
    assert net.skip == set()


def test_main_saves_partial_report_and_restores_skip_on_interrupt(tmp_path, monkeypatch):
    instances = _mock_network(monkeypatch, interrupt_at=4)
    source = _image(tmp_path / "frame.png", value=20)
    out = tmp_path / "new" / "interrupted.md"
    result = ablate.main([
        "--input", str(source), "--device", "cpu", "--out", str(out),
        "--max-evals", "60", "--top-k", "1",
    ])
    assert result == 130
    report = _strict_report(out.with_suffix(".json"))
    assert report["status"] == "interrupted"
    assert report["stop_reason"]
    assert report["progress"]["unique_evaluations"] == 2
    assert len(report["evaluations"]) == 2
    assert out.is_file()
    assert instances[0].skip == set()


@pytest.mark.parametrize("arguments", [
    ["--max-evals", "59"],
    ["--keep-ratio", "0.01"],
    ["--device", "invalid-device"],
])
def test_main_rejects_invalid_search_before_loading_network(tmp_path, monkeypatch, arguments):
    instances = _mock_network(monkeypatch)
    source = _image(tmp_path / "frame.png")
    with pytest.raises(SystemExit) as error:
        ablate.main(["--input", str(source), "--out", str(tmp_path / "report.md"), *arguments])
    assert error.value.code == 2
    assert not instances


def test_main_does_not_overwrite_an_input_image(tmp_path, monkeypatch):
    instances = _mock_network(monkeypatch)
    source = _image(tmp_path / "frame.png")
    original = source.read_bytes()
    with pytest.raises(SystemExit) as error:
        ablate.main(["--input", str(source), "--out", str(source)])
    assert error.value.code == 2
    assert source.read_bytes() == original
    assert not instances


def test_main_reports_model_failure_and_preserves_completed_results(tmp_path, monkeypatch):
    instances = _mock_network(monkeypatch)
    source = _image(tmp_path / "frame.png", value=20)
    output = tmp_path / "report.md"
    fake_call = ablate.NRNet.__call__

    def failing_call(net, frame, controls):
        if net.calls == 3:
            raise RuntimeError("simulated inference failure")
        return fake_call(net, frame, controls)

    monkeypatch.setattr(ablate.NRNet, "__call__", failing_call)
    assert ablate.main(["--input", str(source), "--out", str(output), "--device", "cpu"]) == 1
    report = _strict_report(output.with_suffix(".json"))
    assert report["status"] == "error"
    assert "simulated inference failure" in report["error"]
    assert len(report["evaluations"]) == 2
    assert instances[0].skip == set()
