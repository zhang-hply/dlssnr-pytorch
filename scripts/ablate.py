"""Search for block-pruned networks that approximate the full network within a MAC budget.

    python scripts/ablate.py --input frames/ --keep-ratio 0.8 --out out/ablation.md
    python scripts/ablate.py --input frame.png --reference nvidia.png --max-evals 400

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

    def run(self, evaluate: Callable[[tuple[int, ...]], dict], on_result=None) -> None:
        def measure(blocks):
            key = tuple(sorted(set(blocks)))
            if key in self.results:
                return self.results[key]
            if len(self.results) >= self.config.max_evals:
                raise RuntimeError("internal search evaluation budget exceeded")
            started = time.perf_counter()
            metrics = evaluate(key)
            mse = float(metrics["mse"])
            if not math.isfinite(mse) or mse < 0:
                raise ValueError(f"invalid MSE for blocks {key}: {mse}")
            saved = sum(self.candidates[block].macs for block in key)
            remaining = self.total_macs - saved
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
        for block in self.candidates:
            measure((block,))
        ranking = sorted(self.candidates, key=lambda block: (
            self.results[(block,)]["mse"] / self.candidates[block].macs,
            self.results[(block,)]["mse"], block))

        # Measure a feasible fallback before spending the remaining budget on alternative paths.
        self.phase = "seed"
        seed, remaining = [], self.total_macs
        for block in ranking:
            seed.append(block)
            remaining -= self.candidates[block].macs
            if remaining <= self.total_macs * self.config.keep_ratio:
                break
        measure(seed)

        self.phase = "beam"
        beam = [()]
        while beam:
            children = set()
            for parent in beam:
                additions = [block for block in ranking if block not in parent][:self.config.expand_top_k]
                children.update(tuple(sorted((*parent, block))) for block in additions)
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
    def __init__(self, net, cases: list[ImageCase], controls: Controls, candidates):
        self.net, self.cases, self.controls, self.candidates = net, cases, controls, candidates
        self.forward_calls = {"baseline": 0, "search": 0, "ssim": 0}

    def baseline(self, on_image=None) -> dict:
        self.net.skip = set()
        reference_error, count = 0.0, 0
        for case in self.cases:
            self.forward_calls["baseline"] += 1
            case.full = self.net(case.frame, self.controls)
            if case.reference is not None:
                reference_error += image_mse(case.full, case.reference) * case.frame.size
                count += case.frame.size
            if on_image is not None:
                on_image()
        result = {"mse": 0.0, "psnr_db": math.inf}
        if count:
            result.update(reference_mse=reference_error / count,
                          reference_psnr_db=psnr_from_mse(reference_error / count))
        return result

    def __call__(self, blocks: tuple[int, ...]) -> dict:
        self.net.skip = {i for block in blocks for i in self.candidates[block].indices}
        squared_error, reference_error, count = 0.0, 0.0, 0
        per_image = []
        try:
            for case in self.cases:
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
        finally:
            self.net.skip = set()
        result = {"mse": squared_error / count, "per_image": per_image}
        if self.cases[0].reference is not None:
            result.update(reference_mse=reference_error / count,
                          reference_psnr_db=psnr_from_mse(reference_error / count))
        return result

    def supplement_ssim(self, row: dict) -> None:
        # SSIM's reflection padding needs both image dimensions greater than five.
        if min(self.cases[0].frame.shape[:2]) <= 5:
            row["ssim"] = None
            if self.cases[0].reference is not None:
                row["reference_ssim"] = None
            return
        self.net.skip = set(row["layer_indices"])
        full_scores, reference_scores = [], []
        try:
            for case, image_row in zip(self.cases, row["per_image"]):
                self.forward_calls["ssim"] += 1
                output = self.net(case.frame, self.controls)
                score = ssim(output, case.full)
                full_scores.append(score)
                image_row["ssim"] = score
                if case.reference is not None:
                    score = ssim(output, case.reference)
                    reference_scores.append(score)
                    image_row["reference_ssim"] = score
        finally:
            self.net.skip = set()
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="one image or a directory of same-size images")
    parser.add_argument("--reference", help="optional reference image or directory with matching filenames")
    parser.add_argument("--weights", default="weights.safetensors")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="out/ablation.md", help="Markdown report; sibling .json stores all results")
    parser.add_argument("--keep-ratio", type=float, default=0.8)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--expand-top-k", type=int, default=6)
    parser.add_argument("--max-evals", type=int, default=400,
                        help="maximum unique combinations; excludes baseline and SSIM replays")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        if args.top_k < 1:
            raise ValueError("--top-k must be positive")
        cases = load_inputs(args.input, args.reference)
        height, width = cases[0].frame.shape[:2]
        plan = build_plan(width, height)
        candidates = units(plan)
        config = SearchConfig(args.keep_ratio, args.beam_width, args.expand_top_k, args.max_evals)
        search = BlockSearch(candidates, sum(layer_macs(plan).values()), config)
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
        print(f"[{len(state.results):3d}/{config.max_evals}] {state.phase:<13} "
              f"keep={row['keep_ratio'] * 100:6.2f}% PSNR={row['psnr_db']:6.2f} dB "
              f"blocks={row['blocks']}", flush=True)
        checkpoint()

    exit_code = 0
    try:
        search.phase = "baseline"
        search.stop_reason = "running"
        checkpoint()
        net = NRNet(Weights.load(str(Path(args.weights).expanduser())), width, height).to(device)
        evaluator = ModelEvaluator(net, cases, controls, candidates)
        print(f"{width}x{height}, {len(cases)} image(s), {len(candidates)} skippable blocks; "
              f"full={search.total_macs / 1e9:.3f} GMAC, target <= {config.keep_ratio * 100:.2f}%", flush=True)
        metadata["baseline"] = evaluator.baseline(checkpoint)
        checkpoint()
        search.run(evaluator, on_result)
        search.phase = "ssim"
        for row in search.best_results(args.top_k):
            evaluator.supplement_ssim(row)
            checkpoint()
        search.phase = "done"
        status = "completed"
    except KeyboardInterrupt:
        status, search.stop_reason, exit_code = "interrupted", "interrupted", 130
        print("Interrupted; saving completed evaluations.", file=sys.stderr)
    except Exception as error:
        status, search.stop_reason, exit_code = "error", "error", 1
        error_message = f"{type(error).__name__}: {error}"
        print(error_message, file=sys.stderr)
    finally:
        if net is not None:
            net.skip = set()
        try:
            checkpoint()
        except OSError as error:
            print(f"could not save report: {error}", file=sys.stderr)
            exit_code = exit_code or 1
    print(f"Reports: {output} and {output.with_suffix('.json')}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
