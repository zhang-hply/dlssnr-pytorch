"""Search depth/width-pruned networks that approximate the full model within a MAC budget.

    python scripts/ablate.py --input frames/ --keep-ratio 0.8 --out out/ablation.md
    python scripts/ablate.py --input frame.png --reference nvidia.png --max-evals 400
    python scripts/ablate.py --input frames/ --method compare --out out/pruning_compare.md

Directory inputs are read in filename order, non-recursively, and must have the same dimensions.
References are optional and never affect ranking. One evaluation runs a combination on every input;
the baseline and final Top-K SSIM replays are counted separately. This is a heuristic search, not an
exhaustive one. MACs count matrix multiply-adds, not elementwise work or measured latency.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch

from dlssnr.ablation import BlockUnit, layer_macs, units
from dlssnr.io import Controls, load_image
from dlssnr.metrics import ssim
from dlssnr.model import NRNet
from dlssnr.plan import build_plan
from dlssnr.runtime import resolve_device
from dlssnr.weights import Weights


def psnr_from_mse(mse: float) -> float:
    return math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def log(message: str, *, file=None) -> None:
    """Flush progress immediately, including when stdout is redirected to a file."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=file, flush=True)


@dataclass(frozen=True)
class SearchConfig:
    keep_ratio: float = 0.8
    beam_width: int = 2
    expand_top_k: int = 6
    max_evals: int = 400


class BlockSearch:
    """Model-independent deterministic search; only evaluate() performs inference."""

    def __init__(self, candidates: Mapping[int, BlockUnit], total_macs: int, config: SearchConfig):
        if not math.isfinite(config.keep_ratio) or not 0 < config.keep_ratio < 1:
            raise ValueError("--keep-ratio must be between 0 and 1 (exclusive)")
        if config.beam_width < 1 or config.expand_top_k < 1:
            raise ValueError("--beam-width and --expand-top-k must be positive")
        if config.max_evals < len(candidates) + 1:
            raise ValueError(f"--max-evals must be at least {len(candidates) + 1} for singles and a seed")
        if total_macs <= 0 or not candidates or any(unit.macs <= 0 for unit in candidates.values()):
            raise ValueError("the network and all candidate blocks must have positive MACs")
        if any(block != unit.block for block, unit in candidates.items()):
            raise ValueError("candidate keys must match block numbers")
        available = sum(unit.macs for unit in candidates.values())
        if available > total_macs:
            raise ValueError("candidate MACs exceed the full network MACs")
        if total_macs - available > total_macs * config.keep_ratio:
            raise ValueError("the skippable blocks cannot reach the requested MAC budget")
        self.candidates = dict(sorted(candidates.items()))
        self.total_macs, self.config = total_macs, config
        self.results: dict[tuple[int, ...], dict] = {}
        self.stop_reason = "not_started"
        self.phase = "not_started"

    def best_results(self, top_k: int = 10) -> list[dict]:
        feasible = (row for row in self.results.values() if row["feasible"])
        return sorted(feasible, key=lambda row: (
            row["mse"], -row["macs"], len(row["blocks"]), tuple(row["blocks"])))[:top_k]

    def run(self, evaluate: Callable[[tuple[int, ...]], dict], on_result=None, on_progress=None) -> None:
        def measure(blocks):
            key = tuple(sorted(set(blocks)))
            if key in self.results:
                return self.results[key]
            if len(self.results) >= self.config.max_evals:
                raise RuntimeError("internal search evaluation budget exceeded")
            saved = sum(self.candidates[block].macs for block in key)
            remaining = self.total_macs - saved
            if on_progress is not None:
                on_progress(f"[{len(self.results) + 1:3d}/{self.config.max_evals}] {self.phase:<13} "
                            f"starting keep={remaining / self.total_macs * 100:.2f}% blocks={list(key)}")
            started = time.perf_counter()
            metrics = evaluate(key)
            mse = float(metrics["mse"])
            if not math.isfinite(mse) or mse < 0:
                raise ValueError(f"invalid MSE for blocks {key}: {mse}")
            row = dict(metrics)
            row.update(blocks=list(key), layer_indices=sorted(
                i for block in key for i in self.candidates[block].indices),
                macs=remaining, saved_macs=saved, keep_ratio=remaining / self.total_macs,
                feasible=remaining <= self.total_macs * self.config.keep_ratio,
                mse=mse, psnr_db=psnr_from_mse(mse),
                evaluation_seconds=time.perf_counter() - started)
            self.results[key] = row
            if on_result is not None:
                on_result(self, row)
            return row

        self.stop_reason = "running"
        self.phase = "single_blocks"
        if on_progress is not None:
            on_progress(f"Single-block screening: {len(self.candidates)} blocks")
        for block in self.candidates:
            measure((block,))
        ranking = sorted(self.candidates, key=lambda block: (
            self.results[(block,)]["mse"] / self.candidates[block].macs,
            self.results[(block,)]["mse"], block))

        # Measure a feasible fallback before spending the remaining budget on alternative paths.
        self.phase = "seed"
        if on_progress is not None:
            on_progress("Building a feasible seed from single-block sensitivity scores")
        seed, remaining = [], self.total_macs
        for block in ranking:
            seed.append(block)
            remaining -= self.candidates[block].macs
            if remaining <= self.total_macs * self.config.keep_ratio:
                break
        measure(seed)

        self.phase = "beam"
        beam = [()]
        depth = 0
        while beam:
            depth += 1
            children = set()
            for parent in beam:
                additions = [block for block in ranking if block not in parent][:self.config.expand_top_k]
                children.update(tuple(sorted((*parent, block))) for block in additions)
            if on_progress is not None:
                new_count = sum(key not in self.results for key in children)
                on_progress(f"Beam depth={depth}: paths={len(beam)}, combinations={len(children)}, "
                            f"new={new_count}, evaluated={len(self.results)}/{self.config.max_evals}")
            next_rows = []
            for key in sorted(children):
                if key not in self.results and len(self.results) >= self.config.max_evals:
                    self.stop_reason = "max_evals"
                    self.phase = "done"
                    return
                row = measure(key)
                if not row["feasible"]:
                    next_rows.append(row)
            next_rows.sort(key=lambda row: (
                row["mse"] / row["saved_macs"], row["mse"], -row["macs"], tuple(row["blocks"])))
            beam = [tuple(row["blocks"]) for row in next_rows[:self.config.beam_width]]
        self.stop_reason = "search_exhausted"
        self.phase = "done"


@dataclass
class ImageCase:
    path: Path
    frame: np.ndarray
    reference_path: Path | None = None
    reference: np.ndarray | None = None
    full: np.ndarray | None = None


def load_inputs(input_path, reference_path=None) -> list[ImageCase]:
    source = Path(input_path).expanduser().resolve()
    if source.is_dir():
        extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        paths = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in extensions)
        if not paths:
            raise ValueError(f"no supported images in {source}")
    elif source.is_file():
        paths = [source]
    else:
        raise ValueError(f"input does not exist: {source}")
    reference = Path(reference_path).expanduser().resolve() if reference_path else None
    if reference is not None:
        if not reference.exists() or reference.is_dir() != source.is_dir():
            raise ValueError("--reference must be an existing file/directory matching --input")
    cases = []
    for path in paths:
        frame = load_image(str(path))
        if cases and frame.shape != cases[0].frame.shape:
            raise ValueError(f"all inputs must have the same dimensions: {path}")
        ref_path = reference / path.name if reference is not None and source.is_dir() else reference
        ref_frame = None
        if ref_path is not None:
            if not ref_path.is_file():
                raise ValueError(f"missing reference image: {ref_path}")
            ref_frame = load_image(str(ref_path))
            if ref_frame.shape != frame.shape:
                raise ValueError(f"input/reference dimensions differ: {path} and {ref_path}")
        cases.append(ImageCase(path, frame, ref_path, ref_frame))
    return cases


def image_mse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"image shapes differ: {a.shape} and {b.shape}")
    difference = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(difference * difference))


class ModelEvaluator:
    def __init__(self, net, cases: list[ImageCase], controls: Controls, candidates, on_progress=None,
                 *, applier=None):
        self.net, self.cases, self.controls, self.candidates = net, cases, controls, candidates
        self.forward_calls = {"baseline": 0, "search": 0, "ssim": 0}
        self.on_progress = on_progress
        self.applier = applier

    def reset(self) -> None:
        if self.applier is None:
            self.net.skip = set()
        else:
            self.applier.reset()

    def _select(self, selection) -> None:
        if self.applier is None:
            self.net.skip = {i for block in selection for i in self.candidates[block].indices}
        else:
            self.applier.apply(selection)

    def _log(self, message: str) -> None:
        if self.on_progress is not None:
            self.on_progress(message)

    def baseline(self, on_image=None) -> dict:
        self.reset()
        reference_error, count = 0.0, 0
        for index, case in enumerate(self.cases, 1):
            self._log(f"Baseline image {index}/{len(self.cases)}: {case.path.name} starting")
            started = time.perf_counter()
            self.forward_calls["baseline"] += 1
            case.full = self.net(case.frame, self.controls)
            if case.reference is not None:
                reference_error += image_mse(case.full, case.reference) * case.frame.size
                count += case.frame.size
            self._log(f"Baseline image {index}/{len(self.cases)}: {case.path.name} "
                      f"done in {time.perf_counter() - started:.2f}s")
            if on_image is not None:
                on_image()
        result = {"mse": 0.0, "psnr_db": math.inf}
        if count:
            result.update(reference_mse=reference_error / count,
                          reference_psnr_db=psnr_from_mse(reference_error / count))
        return result

    def __call__(self, blocks: tuple[int, ...]) -> dict:
        squared_error, reference_error, count = 0.0, 0.0, 0
        per_image = []
        try:
            self._select(blocks)
            for index, case in enumerate(self.cases, 1):
                self._log(f"Search image {index}/{len(self.cases)}: {case.path.name} starting")
                started = time.perf_counter()
                self.forward_calls["search"] += 1
                output = self.net(case.frame, self.controls)
                mse = image_mse(output, case.full)
                squared_error += mse * output.size
                count += output.size
                row = {"input": str(case.path), "mse": mse, "psnr_db": psnr_from_mse(mse)}
                if case.reference is not None:
                    error = image_mse(output, case.reference)
                    reference_error += error * output.size
                    row.update(reference_mse=error, reference_psnr_db=psnr_from_mse(error))
                per_image.append(row)
                self._log(f"Search image {index}/{len(self.cases)}: {case.path.name} "
                          f"MSE={mse:.6f} PSNR={row['psnr_db']:.2f}dB "
                          f"done in {time.perf_counter() - started:.2f}s")
        finally:
            self.reset()
        result = {"mse": squared_error / count, "per_image": per_image}
        if self.cases[0].reference is not None:
            result.update(reference_mse=reference_error / count,
                          reference_psnr_db=psnr_from_mse(reference_error / count))
        return result

    def supplement_ssim(self, row: dict) -> None:
        # SSIM's reflection padding needs both image dimensions greater than five.
        if min(self.cases[0].frame.shape[:2]) <= 5:
            self._log("SSIM skipped: image dimensions must exceed five pixels")
            row["ssim"] = None
            if self.cases[0].reference is not None:
                row["reference_ssim"] = None
            return
        full_scores, reference_scores = [], []
        try:
            if self.applier is None:
                self.net.skip = set(row["layer_indices"])
            else:
                from dlssnr.pruning import PruningSpec
                self._select(PruningSpec.from_dict(row["pruning_spec"]))
            for index, (case, image_row) in enumerate(zip(self.cases, row["per_image"]), 1):
                self._log(f"SSIM image {index}/{len(self.cases)}: {case.path.name} starting")
                started = time.perf_counter()
                self.forward_calls["ssim"] += 1
                output = self.net(case.frame, self.controls)
                score = ssim(output, case.full)
                full_scores.append(score)
                image_row["ssim"] = score
                if case.reference is not None:
                    score = ssim(output, case.reference)
                    reference_scores.append(score)
                    image_row["reference_ssim"] = score
                self._log(f"SSIM image {index}/{len(self.cases)}: {case.path.name} "
                          f"SSIM={image_row['ssim']:.4f} done in {time.perf_counter() - started:.2f}s")
        finally:
            self.reset()
        row["ssim"] = float(np.mean(full_scores))
        if reference_scores:
            row["reference_ssim"] = float(np.mean(reference_scores))


def json_safe(value):
    """Use strings for perfect-match PSNR so the output remains standard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_text(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_report(path: Path, payload: dict) -> None:
    atomic_text(path.with_suffix(".json"), json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
    metadata, progress = payload["metadata"], payload["progress"]
    lines = ["# Block pruning search", "", f"Status: {payload['status']}; stop reason: {payload['stop_reason']}",
             "", "Heuristic search: the best measured result is not guaranteed to be globally optimal.",
             "MACs count matrix multiply-adds, excluding elementwise operations and memory transfers.",
             "Scores compare final 8-bit outputs; MAC savings are not measured latency savings.", "",
             f"Input: {metadata['width']}x{metadata['height']}, {len(metadata['inputs'])} image(s)",
             f"Weights: `{metadata['weights']}`; device: `{metadata['device']}`",
             f"Controls: `{json.dumps(metadata['controls'], sort_keys=True)}`",
             f"Search configuration: `{json.dumps(metadata['search'], sort_keys=True)}`",
             f"Full network: {metadata['total_macs'] / 1e9:.6f} GMAC; "
             f"target remaining: <= {metadata['search']['keep_ratio'] * 100:.2f}%",
             f"Evaluations: {progress['unique_evaluations']}; phase: {progress['phase']}; "
             f"elapsed: {progress['elapsed_seconds']:.1f} s",
             f"Forward calls: baseline={progress['forward_calls']['baseline']}, "
             f"search={progress['forward_calls']['search']}, SSIM replays={progress['forward_calls']['ssim']}", "",
             "Inputs:", *[f"- `{item['input']}`" + (
                 f" (reference: `{item['reference']}`)" if item['reference'] else "")
                 for item in metadata["inputs"]], ""]
    if payload.get("error"):
        lines += [f"Error: {payload['error']}", ""]
    if metadata.get("baseline") is not None:
        lines += [f"Baseline: `{json.dumps(json_safe(metadata['baseline']), sort_keys=True)}`", ""]
    if payload["best"] is None:
        lines += ["No feasible combination has been evaluated yet.", ""]
    else:
        best = payload["best"]
        lines += ["## Best measured combination", "", f"Dropped blocks: `{best['blocks']}`",
                  f"Remaining: {best['macs'] / 1e9:.6f} GMAC ({best['keep_ratio'] * 100:.4f}%)", "",
                  "```python", f"drop_blocks = {{{', '.join(map(str, best['blocks']))}}}",
                  "net.skip = {s.index for s in net.plan if s.block in drop_blocks}", "```", "",
                  f"Layer indices: `{best['layer_indices']}`", ""]
    lines += ["## Top feasible combinations", "",
              "| Blocks dropped | GMAC remaining | Remaining % | MSE vs full | PSNR vs full | SSIM vs full | PSNR vs reference | SSIM vs reference |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]

    def number(value, digits=4):
        return "N/A" if value is None else f"{value:.{digits}f}"

    for row in payload["top_results"]:
        lines.append(f"| {row['blocks']} | {row['macs'] / 1e9:.6f} | {row['keep_ratio'] * 100:.4f} | "
                     f"{row['mse']:.6f} | {number(row['psnr_db'], 2)} | {number(row.get('ssim'))} | "
                     f"{number(row.get('reference_psnr_db'), 2)} | {number(row.get('reference_ssim'))} |")
    lines += ["", "## Single-block sensitivity", "",
              "| Block | Family | MACs saved % | MSE vs full | PSNR vs full | PSNR vs reference |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    families = {unit["block"]: unit["family"] for unit in metadata["candidates"]}
    for row in payload["single_block_results"]:
        block = row["blocks"][0]
        lines.append(f"| {block} | {families[block]} | {100 * row['saved_macs'] / metadata['total_macs']:.4f} | "
                     f"{row['mse']:.6f} | {number(row['psnr_db'], 2)} | {number(row.get('reference_psnr_db'), 2)} |")
    lines += ["", "All evaluated combinations, per-image scores and run metadata are in "
              f"`{path.with_suffix('.json').name}`.",
              "SSIM is calculated only for final Top-K results; N/A means pending or unavailable "
              "(image dimensions must exceed five pixels).", ""]
    atomic_text(path, "\n".join(lines))


def describe_pruning(raw: dict) -> str:
    """Compact display; original retained indices remain in the JSON spec."""
    descriptions = []
    if raw["dropped_blocks"]:
        descriptions.append(f"drop={raw['dropped_blocks']}")
    for width in raw["widths"]:
        axes = []
        if "mlp" in width:
            axes.append(f"MLP={len(width['mlp'])}x{len(width['mlp'][0])}")
        if "heads" in width:
            axes.append(f"heads={len(width['heads'])}")
        descriptions.append(f"b{width['block']}({', '.join(axes)})")
    return "; ".join(descriptions) or "full model"


def write_structured_report(path: Path, payload: dict) -> None:
    atomic_text(path.with_suffix(".json"), json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
    metadata, progress = payload["metadata"], payload["progress"]
    lines = ["# Depth and width pruning search", "",
             f"Status: {payload['status']}; stop reason: {payload['stop_reason']}", "",
             "Heuristic search: these are the best measured structures, not a guarantee of global optimality.",
             "Scores compare final uint8 outputs against one shared full-network baseline.",
             "Matrix MACs exclude elementwise work and memory transfers; savings are not measured latency.", "",
             f"Input: {metadata['width']}x{metadata['height']}, {len(metadata['inputs'])} image(s)",
             f"Weights: `{metadata['weights']}`; SHA256: `{metadata.get('weights_sha256', 'pending')}`",
             f"Controls: `{json.dumps(metadata['controls'], sort_keys=True)}`",
             f"Search configuration: `{json.dumps(metadata['search'], sort_keys=True)}`",
             f"Full model: {metadata['total_macs'] / 1e9:.6f} GMAC; "
             f"target remaining: <= {metadata['search']['keep_ratio'] * 100:.2f}%", "",
             f"Shared unique evaluations: {progress['unique_evaluations']}; "
             f"logical method requests: {progress['logical_evaluations']}; phase: {progress['phase']}",
             f"Forward calls: baseline={progress['forward_calls']['baseline']}, "
             f"search={progress['forward_calls']['search']}, SSIM replays={progress['forward_calls']['ssim']}",
             "Each method has its own max_evals limit. Cross-method cache hits count toward that method's "
             "logical budget but do not repeat inference.", "",
             "Width candidates retain 75% or 50% of MLP hidden dimensions and whole attention heads; "
             "residual stream widths stay fixed. Original retained indices are recorded in pruning_spec.", ""]
    if payload.get("error"):
        lines += [f"Error: {payload['error']}", ""]
    lines += ["## Best measured structure", ""]
    if payload["best"] is None:
        lines += ["No feasible structure has been evaluated yet.", ""]
    else:
        best = payload["best"]
        lines += [describe_pruning(best["pruning_spec"]), "",
                  f"Remaining MACs: {best['keep_ratio'] * 100:.4f}%; "
                  f"MSE={best['mse']:.6f}; PSNR={best['psnr_db']:.2f} dB", ""]
    for name, result in payload["methods"].items():
        stats = result["progress"]
        lines += [f"## {name}", "",
                  f"Stop reason: {result['stop_reason']}; phase: {stats['phase']}; "
                  f"evaluated={stats['unique_evaluations']}/{stats['max_evals']}", "",
                  "| Rank | Structure | GMAC | Remaining % | MSE | PSNR dB | SSIM |",
                  "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
        for rank, row in enumerate(result["top_results"], 1):
            structure = "N/A" if row.get("ssim") is None else f"{row['ssim']:.4f}"
            lines.append(f"| {rank} | {describe_pruning(row['pruning_spec'])} | "
                         f"{row['macs'] / 1e9:.6f} | {row['keep_ratio'] * 100:.4f} | "
                         f"{row['mse']:.6f} | {row['psnr_db']:.2f} | {structure} |")
        if not result["top_results"]:
            lines += ["", "No feasible structure has been evaluated for this method yet."]
        lines += [""]
    lines += [f"All unique measurements, per-image scores and replayable specifications: "
              f"`{path.with_suffix('.json').name}`.",
              "SSIM is replayed once per distinct final Top-K structure across methods. "
              "N/A means pending or unavailable (image dimensions must exceed five pixels).", ""]
    atomic_text(path, "\n".join(lines))


def run_structured_search(args, cases, plan, config, output, device, suite) -> int:
    from dlssnr.distill_data import file_sha256
    from dlssnr.distillation import require_reference
    from dlssnr.width_pruning import PruningApplier, rank_widths

    height, width = cases[0].frame.shape[:2]
    controls = Controls()
    metadata = {
        "inputs": [{"input": str(case.path), "reference": str(case.reference_path)
                    if case.reference_path is not None else None} for case in cases],
        "width": width, "height": height, "weights": str(Path(args.weights).expanduser().resolve()),
        "device": str(device), "torch_version": torch.__version__, "controls": asdict(controls),
        "backend_flags": {key: os.environ.get(key, "1") for key in ("DLSSNR_TRITON", "DLSSNR_COMPILE")},
        "method": args.method, "search": asdict(config), "top_k": args.top_k,
        "total_macs": suite.total_macs, "baseline": None,
        "width_grid": [1.0, 0.75, 0.5],
        "width_ranking": "fixed teacher L2 incoming/outgoing products; heads use V/output weights; "
                         "ties by original index",
    }
    started = time.perf_counter()
    status, error_message, exit_code = "running", None, 0
    net, evaluator = None, None

    def checkpoint(*_):
        top = suite.best_results(args.top_k)
        phase = suite.phase
        if phase == "search" and suite.active_method is not None:
            phase = f"{suite.active_method}/{suite.methods[suite.active_method].phase}"
        payload = {
            "schema_version": 2, "status": status, "stop_reason": suite.stop_reason,
            "metadata": metadata, "error": error_message,
            "progress": {"unique_evaluations": len(suite.cache.results),
                         "logical_evaluations": sum(len(search.results) for search in suite.methods.values()),
                         "phase": phase, "elapsed_seconds": time.perf_counter() - started,
                         "forward_calls": dict(evaluator.forward_calls) if evaluator is not None
                         else {"baseline": 0, "search": 0, "ssim": 0}},
            "best": top[0] if top else None, "top_results": top,
            "methods": suite.snapshot(args.top_k), "evaluations": list(suite.cache.results.values()),
        }
        write_structured_report(output, payload)

    def on_result(search, row):
        log(f"{search.method} [{len(search.results)}/{config.max_evals}] {search.phase}: "
            f"keep={row['keep_ratio']:.4%} MSE={row['mse']:.6f} "
            f"PSNR={row['psnr_db']:.2f}dB feasible={'yes' if row['feasible'] else 'no'} "
            f"shared_evaluations={len(suite.cache.results)}; {describe_pruning(row['pruning_spec'])}")
        checkpoint()

    try:
        suite.phase, suite.stop_reason = "baseline", "running"
        checkpoint()
        require_reference()
        metadata["weights_sha256"] = file_sha256(metadata["weights"])
        log(f"Loading reference network on {device}: {metadata['weights']}")
        net = NRNet(Weights.load(metadata["weights"]), width, height).to(device)
        require_reference(net)
        log("Ranking original teacher MLP channels and attention heads")
        suite.set_rankings(rank_widths(net))
        evaluator = ModelEvaluator(net, cases, controls, units(plan), on_progress=log,
                                   applier=PruningApplier(net))
        log(f"{width}x{height}, {len(cases)} image(s); full={suite.total_macs / 1e9:.3f} GMAC; "
            f"target <= {config.keep_ratio:.2%}; budget={config.max_evals} per method")
        metadata["baseline"] = evaluator.baseline(checkpoint)
        checkpoint()
        suite.run(evaluator, on_result, log)
        suite.phase = "ssim"
        replay = suite.ssim_results(args.top_k)
        log(f"Scoring SSIM for {len(replay)} distinct final structures across methods")
        for index, row in enumerate(replay, 1):
            log(f"SSIM structure {index}/{len(replay)}: {describe_pruning(row['pruning_spec'])}")
            evaluator.supplement_ssim(row)
            checkpoint()
        suite.phase, status = "done", "completed"
    except KeyboardInterrupt:
        status, suite.stop_reason, exit_code = "interrupted", "interrupted", 130
        log("Interrupted; saving completed pruning evaluations.", file=sys.stderr)
    except Exception as error:
        status, suite.stop_reason, exit_code = "error", "error", 1
        error_message = f"{type(error).__name__}: {error}"
        log(error_message, file=sys.stderr)
    finally:
        try:
            if evaluator is not None:
                evaluator.reset()
            elif net is not None:
                net.skip = set()
        except Exception as error:
            log(f"could not restore full model: {error}", file=sys.stderr)
            exit_code = exit_code or 1
        try:
            checkpoint()
        except OSError as error:
            log(f"could not save report: {error}", file=sys.stderr)
            exit_code = exit_code or 1
    log(f"Finished: status={status}, reason={suite.stop_reason}, "
        f"unique_evaluations={len(suite.cache.results)}, elapsed={time.perf_counter() - started:.2f}s")
    log(f"Reports: {output} and {output.with_suffix('.json')}")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="one image or a directory of same-size images")
    parser.add_argument("--reference", help="optional reference image or directory with matching filenames")
    parser.add_argument("--weights", default="weights.safetensors")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="out/ablation.md", help="Markdown report; sibling .json stores all results")
    parser.add_argument("--keep-ratio", type=float, default=0.8)
    parser.add_argument("--method", choices=("depth", "width", "hybrid", "compare"), default="depth",
                        help="depth preserves legacy whole-block search; compare runs depth, width and hybrid")
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--expand-top-k", type=int, default=6)
    parser.add_argument("--max-evals", type=int, default=400,
                        help="maximum unique combinations per method; excludes baseline and SSIM replays")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        if args.top_k < 1:
            raise ValueError("--top-k must be positive")
        log(f"Loading input images: {args.input}")
        cases = load_inputs(args.input, args.reference)
        height, width = cases[0].frame.shape[:2]
        plan = build_plan(width, height)
        candidates = units(plan)
        config = SearchConfig(args.keep_ratio, args.beam_width, args.expand_top_k, args.max_evals)
        suite = None
        if args.method == "depth":
            search = BlockSearch(candidates, sum(layer_macs(plan).values()), config)
        else:
            from dlssnr.pruning_search import SearchSuite
            suite = SearchSuite(plan, config, args.method, BlockSearch)
        output = Path(args.out).expanduser().resolve()
        if output.suffix.lower() == ".json":
            raise ValueError("--out must name a Markdown report, not a .json file")
        protected = {case.path.resolve() for case in cases}
        protected.update(case.reference_path.resolve() for case in cases if case.reference_path is not None)
        protected.add(Path(args.weights).expanduser().resolve())
        for target in (output, output.with_suffix(".json")):
            if target in protected or target.is_dir():
                raise ValueError(f"report would overwrite an input or directory: {target}")
        output.parent.mkdir(parents=True, exist_ok=True)
        for target in (output, output.with_suffix(".json")):
            if target.exists() and not os.access(target, os.W_OK):
                raise ValueError(f"report is not writable: {target}")
        with tempfile.TemporaryFile(dir=output.parent):
            pass
        device = resolve_device(args.device)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))

    if suite is not None:
        return run_structured_search(args, cases, plan, config, output, device, suite)

    controls = Controls()
    metadata = {
        "inputs": [{"input": str(case.path), "reference": str(case.reference_path)
                    if case.reference_path is not None else None} for case in cases],
        "width": width, "height": height, "weights": str(Path(args.weights).expanduser().resolve()),
        "device": str(device), "torch_version": torch.__version__, "controls": asdict(controls),
        "backend_flags": {key: os.environ.get(key, "1") for key in ("DLSSNR_TRITON", "DLSSNR_COMPILE")},
        "search": asdict(config), "top_k": args.top_k, "total_macs": search.total_macs,
        "candidates": [asdict(unit) for unit in candidates.values()], "baseline": None,
    }
    started = time.perf_counter()
    status, error_message = "running", None
    net, evaluator = None, None

    def checkpoint(*_):
        top = search.best_results(args.top_k)
        payload = {
            "schema_version": 1, "status": status, "stop_reason": search.stop_reason,
            "metadata": metadata, "error": error_message,
            "progress": {"unique_evaluations": len(search.results), "phase": search.phase,
                         "elapsed_seconds": time.perf_counter() - started,
                         "forward_calls": dict(evaluator.forward_calls) if evaluator is not None else
                         {"baseline": 0, "search": 0, "ssim": 0}},
            "best": top[0] if top else None, "top_results": top,
            "single_block_results": [row for row in search.results.values() if len(row["blocks"]) == 1],
            "evaluations": list(search.results.values()),
        }
        write_report(output, payload)

    def on_result(state, row):
        best = state.best_results(1)
        best_note = (f" best_PSNR={best[0]['psnr_db']:.2f}dB" if best else "")
        if best and best[0] is row:
            best_note += " [new best]"
        log(f"[{len(state.results):3d}/{config.max_evals}] {state.phase:<13} "
            f"done keep={row['keep_ratio'] * 100:6.2f}% "
            f"MACs={row['macs'] / 1e9:.3f}G MSE={row['mse']:.6f} PSNR={row['psnr_db']:6.2f}dB "
            f"feasible={'yes' if row['feasible'] else 'no'} "
            f"eval={row['evaluation_seconds']:.2f}s blocks={row['blocks']}{best_note}")
        checkpoint()

    exit_code = 0
    try:
        search.phase = "baseline"
        search.stop_reason = "running"
        checkpoint()
        log(f"Loading weights and building network on {device}: {metadata['weights']}")
        loading_started = time.perf_counter()
        net = NRNet(Weights.load(str(Path(args.weights).expanduser())), width, height).to(device)
        log(f"Network ready in {time.perf_counter() - loading_started:.2f}s")
        evaluator = ModelEvaluator(net, cases, controls, candidates, on_progress=log)
        log(f"{width}x{height}, {len(cases)} image(s), {len(candidates)} skippable blocks; "
            f"full={search.total_macs / 1e9:.3f} GMAC, target <= {config.keep_ratio * 100:.2f}%")
        log("Running full-network baseline (first inference may compile kernels)")
        metadata["baseline"] = evaluator.baseline(checkpoint)
        log("Full-network baseline complete")
        checkpoint()
        search.run(evaluator, on_result, on_progress=log)
        search.phase = "ssim"
        top_results = search.best_results(args.top_k)
        log(f"Search stopped: {search.stop_reason}; scoring SSIM for {len(top_results)} best combinations")
        for rank, row in enumerate(top_results, 1):
            log(f"SSIM combination {rank}/{len(top_results)}: blocks={row['blocks']}")
            evaluator.supplement_ssim(row)
            checkpoint()
        search.phase = "done"
        status = "completed"
    except KeyboardInterrupt:
        status, search.stop_reason, exit_code = "interrupted", "interrupted", 130
        log("Interrupted; saving completed evaluations.", file=sys.stderr)
    except Exception as error:
        status, search.stop_reason, exit_code = "error", "error", 1
        error_message = f"{type(error).__name__}: {error}"
        log(error_message, file=sys.stderr)
    finally:
        if net is not None:
            net.skip = set()
        try:
            checkpoint()
        except OSError as error:
            log(f"could not save report: {error}", file=sys.stderr)
            exit_code = exit_code or 1
    log(f"Finished: status={status}, reason={search.stop_reason}, "
        f"evaluated={len(search.results)}/{config.max_evals}, elapsed={time.perf_counter() - started:.2f}s")
    best = search.best_results(1)
    if best:
        log(f"Best measured: keep={best[0]['keep_ratio'] * 100:.4f}% "
            f"PSNR={best[0]['psnr_db']:.2f}dB blocks={best[0]['blocks']}")
    log(f"Reports: {output} and {output.with_suffix('.json')}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
