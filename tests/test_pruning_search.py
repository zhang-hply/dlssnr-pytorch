"""Search regressions use structural MACs and fake image scores, never GPU weights."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from dlssnr.ablation import layer_macs, units
from dlssnr.plan import build_plan
from dlssnr.pruning import PruningSpec, WidthSpec, validate_spec
from dlssnr.pruning_search import EvaluationCache, SearchSuite, WidthSearch, search_sites
from scripts import ablate


@pytest.fixture
def small_plan():
    # A single-head block and a two-head block retain the actual grouped MAC
    # formulas, while keeping exhaustive state-space checks small.
    return [spec for spec in build_plan(64, 64) if spec.block in (1, 5)]


def evaluator(plan, calls):
    def evaluate(spec):
        assert spec not in calls, "shared inference must not repeat a canonical structure"
        calls.append(spec)
        remaining = validate_spec(plan, spec)["keep_ratio"]
        # Joint structure has a deliberately non-additive measured error.
        mse = (1 - remaining) ** 2 * 100 + len(spec.dropped_blocks)
        if len(spec.widths) > 1:
            mse *= .25
        return {"mse": mse}
    return evaluate


def test_1080_catalog_has_expected_shallow_screening_counts():
    plan = build_plan(1920, 1080)
    assert {name: len(search_sites(plan, name)) for name in ("depth", "width", "hybrid")} == {
        "depth": 59, "width": 112, "hybrid": 171}
    sites = search_sites(plan, "width")
    assert not any(site.block == 1 and site.kind == "head" for site in sites)
    assert next(site.levels for site in sites if site.block == 5 and site.kind == "head") == (1,)
    assert next(site.levels for site in sites if site.block == 23 and site.kind == "mlp") == (192, 128)


@pytest.mark.parametrize("method,minimum", [("width", 113), ("hybrid", 172)])
def test_first_level_screening_and_one_seed_fit_exact_minimum_budget(method, minimum):
    plan = build_plan(1920, 1080)
    with pytest.raises(ValueError, match=f"at least {minimum}"):
        WidthSearch(plan, ablate.SearchConfig(max_evals=minimum - 1), method)
    search = WidthSearch(plan, ablate.SearchConfig(max_evals=minimum), method)
    cache = EvaluationCache(plan, .8)
    calls = []
    cache.evaluate = evaluator(plan, calls)
    search.run(cache)
    assert len(search.results) == minimum
    assert len(calls) == minimum
    assert search.best_results(1)[0]["keep_ratio"] <= .8
    first = list(search.results)[:minimum - 1]
    for site, spec in zip(search.sites, first):
        assert spec == search.next_spec(PruningSpec(), site)
    assert search.stop_reason == "max_evals"


def test_unreachable_width_target_is_rejected_without_inference():
    with pytest.raises(ValueError, match="minimum remaining MAC ratio"):
        WidthSearch(build_plan(1920, 1080), ablate.SearchConfig(keep_ratio=.5), "width")


def test_teacher_rankings_are_nested_and_whole_override_has_one_identity(small_plan):
    search = WidthSearch(small_plan, ablate.SearchConfig(max_evals=50), "hybrid")
    ranked = {block: {"mlp": tuple(tuple(reversed(group)) for group in rank["mlp"]),
                      "heads": tuple(reversed(rank["heads"]))}
              for block, rank in search.rankings.items()}
    search.set_rankings(ranked)
    mlp = next(site for site in search.sites if site.block == 5 and site.kind == "mlp")
    head = next(site for site in search.sites if site.block == 5 and site.kind == "head")
    drop = next(site for site in search.sites if site.block == 5 and site.kind == "whole")
    first = search.next_spec(PruningSpec(), mlp)
    second = search.next_spec(first, mlp)
    assert len(first.widths[0].mlp[0]) == 96
    assert first.widths[0].mlp[0][0] == 32
    assert len(second.widths[0].mlp[0]) == 64
    assert all(set(b).issubset(a) for a, b in zip(first.widths[0].mlp, second.widths[0].mlp))
    joint = search.next_spec(second, head)
    assert joint.widths[0].heads == (1,)
    assert search.cost(joint) < search.cost(second) < search.cost(first) < search.total_macs
    dropped = search.next_spec(joint, drop)
    assert dropped == search.next_spec(PruningSpec(), drop) == PruningSpec((5,))
    assert not dropped.widths
    assert search.next_spec(dropped, head) is None
    assert search.cost(dropped) == search.total_macs - units(small_plan)[5].macs
    cache = EvaluationCache(small_plan, .8)
    calls = []
    cache.evaluate = evaluator(small_plan, calls)
    assert cache(dropped) is cache(PruningSpec((5,)))
    assert len(calls) == 1


def test_invalid_teacher_rankings_are_rejected(small_plan):
    search = WidthSearch(small_plan, ablate.SearchConfig(max_evals=50), "width")
    bad = dict(search.rankings)
    bad[1] = {"mlp": ((0,) * 128,), "heads": (0,)}
    with pytest.raises(ValueError, match="invalid teacher width ranking"):
        search.set_rankings(bad)


def test_compare_has_independent_logical_budgets_but_one_inference_cache(small_plan):
    suite = SearchSuite(small_plan, ablate.SearchConfig(max_evals=6), "compare", ablate.BlockSearch)
    calls = []
    suite.run(evaluator(small_plan, calls))
    logical = sum(len(search.results) for search in suite.methods.values())
    assert len(calls) == len(suite.cache.results) < logical
    for method, search in suite.methods.items():
        assert len(search.results) <= 6
        assert search.best_results(1)[0]["feasible"]
        for spec, row in search.results.items():
            assert row is suite.cache.results[spec]
            assert "blocks" not in row
            if method == "depth":
                assert not spec.widths
            elif method == "width":
                assert not spec.dropped_blocks
    assert sum(item["progress"]["unique_evaluations"] for item in suite.snapshot().values()) == logical


def test_depth_adapter_keeps_original_evaluation_order_and_best(small_plan):
    config = ablate.SearchConfig(max_evals=20, beam_width=3, expand_top_k=4)
    original = ablate.BlockSearch(units(small_plan), sum(layer_macs(small_plan).values()), config)
    original_calls = []

    def measure(blocks):
        original_calls.append(blocks)
        return {"mse": float(sum(blocks) ** 2)}

    original.run(measure)
    suite = SearchSuite(small_plan, config, "depth", ablate.BlockSearch)
    adapted_calls = []

    def adapted(spec):
        adapted_calls.append(spec.dropped_blocks)
        return {"mse": float(sum(spec.dropped_blocks) ** 2)}

    suite.run(adapted)
    assert adapted_calls == original_calls
    assert suite.best_results(1)[0]["pruning_spec"]["dropped_blocks"] == original.best_results(1)[0]["blocks"]


def test_search_and_ssim_selection_are_deterministic_and_deduplicated(small_plan):
    runs = []
    for _ in range(2):
        suite = SearchSuite(small_plan, ablate.SearchConfig(max_evals=50, expand_top_k=6),
                            "compare", ablate.BlockSearch)
        calls = []
        suite.run(evaluator(small_plan, calls))
        replay = suite.ssim_results(3)
        specs = [PruningSpec.from_dict(row["pruning_spec"]) for row in replay]
        assert len(specs) == len(set(specs))
        expected = {PruningSpec.from_dict(row["pruning_spec"])
                    for search in suite.methods.values() for row in search.best_results(3)}
        assert set(specs) == expected
        assert suite.stop_reason == "search_exhausted"
        runs.append((calls, specs))
    assert runs[0] == runs[1]


def test_interrupted_search_preserves_only_successful_measurements(small_plan):
    suite = SearchSuite(small_plan, ablate.SearchConfig(max_evals=20), "width", ablate.BlockSearch)
    attempts = 0

    def interrupted(_spec):
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            raise KeyboardInterrupt
        return {"mse": 2.0}

    with pytest.raises(KeyboardInterrupt):
        suite.run(interrupted)
    assert len(suite.cache.results) == len(suite.methods["width"].results) == 2
    assert suite.active_method == "width"


def _mock_structured_cli(monkeypatch, interrupt_at=None):
    from dlssnr import distillation, width_pruning
    instances = []

    class Network:
        def __init__(self, _weights, width, height):
            self.plan = build_plan(width, height)
            self.skip, self.pruning_spec, self.calls = set(), PruningSpec(), 0
            instances.append(self)

        def to(self, _device):
            return self

        def __call__(self, frame, _controls):
            self.calls += 1
            if self.calls == interrupt_at:
                raise KeyboardInterrupt
            error = 0 if self.pruning_spec == PruningSpec() else 2
            return (frame.astype(np.int16) + error).clip(0, 255).astype(np.uint8)

    class Applier:
        def __init__(self, net):
            self.net = net

        def apply(self, spec):
            self.net.pruning_spec = spec
            self.net.skip = set(validate_spec(self.net.plan, spec)["layer_indices"])

        def reset(self):
            self.apply(PruningSpec())

    def rankings(net):
        return WidthSearch(net.plan, ablate.SearchConfig(), "width").rankings

    monkeypatch.setattr(ablate, "NRNet", Network)
    monkeypatch.setattr(ablate, "Weights", SimpleNamespace(load=lambda _path: object()))
    monkeypatch.setattr(distillation, "require_reference", lambda _net=None: None)
    monkeypatch.setattr(width_pruning, "PruningApplier", Applier)
    monkeypatch.setattr(width_pruning, "rank_widths", rankings)
    return instances


@pytest.mark.parametrize("interrupt_at,exit_code", [(None, 0), (4, 130)])
def test_cli_v2_report_and_partial_state(tmp_path, monkeypatch, interrupt_at, exit_code):
    instances = _mock_structured_cli(monkeypatch, interrupt_at)
    source = tmp_path / "frame.png"
    Image.fromarray(np.full((4, 4, 3), 20, dtype=np.uint8)).save(source)
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"fake teacher identity")
    output = tmp_path / "report.md"
    assert ablate.main(["--input", str(source), "--weights", str(weights), "--device", "cpu",
                        "--out", str(output), "--method", "compare", "--max-evals", "172",
                        "--top-k", "1"]) == exit_code
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["schema_version"] == 2
    assert len(report["metadata"]["weights_sha256"]) == 64
    assert set(report["methods"]) == {"depth", "width", "hybrid"}
    assert report["progress"]["forward_calls"]["baseline"] == 1
    assert report["progress"]["unique_evaluations"] == len(report["evaluations"])
    assert instances[0].pruning_spec == PruningSpec()
    assert not instances[0].skip
    assert all("pruning_spec" in row and "blocks" not in row for row in report["evaluations"])
    if interrupt_at is None:
        assert report["status"] == "completed"
        assert report["progress"]["logical_evaluations"] > len(report["evaluations"])
        assert all(row["best"]["feasible"] for row in report["methods"].values())
        assert "## width" in output.read_text()
    else:
        assert report["status"] == "interrupted"
        assert len(report["evaluations"]) == 2
        assert report["methods"]["width"]["progress"]["unique_evaluations"] == 0


def test_structured_evaluator_applies_same_spec_for_ssim_and_restores():
    class Net:
        skip = set()
        spec = PruningSpec()

        def __call__(self, frame, controls):
            return frame + (2 if self.spec.widths else 0)

    net = Net()
    selected = []

    class Applier:
        def apply(self, spec):
            net.spec = spec
            selected.append(spec)

        def reset(self):
            net.spec = PruningSpec()

    image = np.full((8, 8, 3), 20, dtype=np.uint8)
    case = ablate.ImageCase(__import__("pathlib").Path("frame.png"), image)
    score = ablate.ModelEvaluator(net, [case], ablate.Controls(), {}, applier=Applier())
    score.baseline()
    spec = PruningSpec(widths=(WidthSpec(1, mlp=(tuple(range(64)),)),))
    row = score(spec)
    row["pruning_spec"] = spec.to_dict()
    score.supplement_ssim(row)
    assert selected == [spec, spec]
    assert row["mse"] == 4
    assert row["ssim"] < 1
    assert net.spec == PruningSpec()
    assert score.forward_calls == {"baseline": 1, "search": 1, "ssim": 1}
