"""Replay a search report's best mask and save full/pruned image pairs."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from .distillation import read_pruning_report, require_reference
from .io import Controls, load_image, save_image
from .metrics import ssim
from .model import NRNet
from .plan import build_plan
from .runtime import clear_caches, resolve_device
from .weights import Weights


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _report_path(path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() in {".md", ".markdown"}:
        path = path.with_suffix(".json")
    if not path.is_file():
        raise ValueError(f"ablation JSON report not found: {path}; a Markdown report needs its sibling JSON")
    return path


def _inputs(metadata: dict, input_path) -> list[Path]:
    if input_path is not None:
        source = Path(input_path).expanduser().resolve()
        if source.is_dir():
            paths = sorted(path for path in source.iterdir()
                           if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS)
        elif source.is_file():
            paths = [source]
        else:
            raise ValueError(f"input does not exist: {source}")
    else:
        entries = metadata.get("inputs", [])
        if not isinstance(entries, list) or any(
                not isinstance(entry, dict) or not isinstance(entry.get("input"), str) for entry in entries):
            raise ValueError("report inputs are invalid; supply --input with an image or directory")
        paths = [Path(entry["input"]).expanduser().resolve() for entry in entries]
    if not paths:
        raise ValueError("no input images found; supply --input with an image or directory")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"recorded input is missing: {path}; use --input to override report paths")
    return paths


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_write(path: Path, writer) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.",
                                         suffix=path.suffix, delete=False) as handle:
            temporary = Path(handle.name)
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _save_manifest(root: Path, manifest: dict) -> None:
    text = json.dumps(_json_safe(manifest), indent=2, allow_nan=False) + "\n"
    _atomic_write(root / "comparison.json", lambda path: path.write_text(text, encoding="utf-8"))


def compare_ablation(report_path, input_path=None, out_dir="out/ablation_comparison",
                     weights_path="weights.safetensors", device="auto", log=print) -> dict:
    """Save baseline, best-mask output and a left-baseline/right-pruned PNG per input.

    With no explicit input, replay the image paths recorded in the report. Input
    dimensions and whole-block safety are checked, but no new 80% budget is imposed:
    the report may have been searched with another keep ratio. Reports are not
    modified and the search is never rerun. A nonempty output directory is rejected.
    """
    report_path = _report_path(report_path)
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict) or not isinstance(report.get("metadata", {}), dict):
        raise ValueError("ablation report and metadata must be JSON objects")
    metadata = report.get("metadata", {})
    paths = _inputs(metadata, input_path)
    frames = [load_image(str(path)) for path in paths]
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("all comparison inputs must have the same dimensions")
    height, width = frames[0].shape[:2]
    if (("width" in metadata and metadata["width"] != width)
            or ("height" in metadata and metadata["height"] != height)):
        # Different raw image sizes can share an identical padded execution plan.
        raise ValueError("comparison input dimensions differ from the ablation report")
    pruning = read_pruning_report(report_path, build_plan(width, height), keep_ratio=1.0)
    settings = metadata.get("controls", {})
    if not isinstance(settings, dict):
        raise ValueError("report controls must be an object")
    try:
        controls = Controls(**settings)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid report controls: {error}") from error
    source = Path(weights_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"weights file does not exist: {source}")
    device = resolve_device(str(device))
    require_reference()
    root = Path(out_dir).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("comparison output must be an empty or new directory")
    root.mkdir(parents=True, exist_ok=True)
    for folder in ("baseline", "pruned", "side_by_side"):
        (root / folder).mkdir()
    manifest = {
        "schema_version": 1, "status": "running", "report": str(report_path), "weights": str(source),
        "device": str(device), "width": width, "height": height, "controls": asdict(controls),
        "dropped_blocks": pruning["blocks"], "layer_indices": pruning["layer_indices"],
        "total_macs": pruning["total_macs"], "macs": pruning["macs"], "keep_ratio": pruning["keep_ratio"],
        "side_by_side_order": ["baseline", "pruned"], "images": [],
    }
    net = None
    started = time.perf_counter()
    squared, elements, structures = 0.0, 0, []
    try:
        _save_manifest(root, manifest)
        log(f"Loading model on {device}; dropped blocks={pruning['blocks']}; "
            f"remaining MACs={pruning['keep_ratio'] * 100:.4f}%")
        net = NRNet(Weights.load(str(source)), width, height).to(device).eval()
        for index, (path, frame) in enumerate(zip(paths, frames), 1):
            log(f"[{index}/{len(paths)}] {path.name}: full-model baseline starting")
            net.skip = set()
            baseline = net(frame, controls)
            log(f"[{index}/{len(paths)}] {path.name}: best-mask inference starting")
            net.skip = set(pruning["layer_indices"])
            pruned = net(frame, controls)
            net.skip = set()
            if baseline.shape != frame.shape or pruned.shape != frame.shape:
                raise ValueError("model outputs must have the same shape as the input image")
            difference = pruned.astype(np.float64) - baseline.astype(np.float64)
            mse = float(np.mean(difference * difference))
            psnr = math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
            structure = ssim(pruned, baseline) if min(width, height) > 5 else None
            filename = f"{index:04d}_{path.stem}.png"
            row = {"input": str(path), "baseline": f"baseline/{filename}", "pruned": f"pruned/{filename}",
                   "side_by_side": f"side_by_side/{filename}", "mse": mse, "psnr_db": psnr, "ssim": structure}
            for name, pixels in (("baseline", baseline), ("pruned", pruned),
                                 ("side_by_side", np.concatenate((baseline, pruned), axis=1))):
                _atomic_write(root / row[name], lambda target, pixels=pixels: save_image(str(target), pixels))
            squared += mse * pruned.size
            elements += pruned.size
            if structure is not None:
                structures.append(structure)
            manifest["images"].append(row)
            manifest["aggregate"] = {"mse": squared / elements,
                                     "psnr_db": math.inf if squared == 0 else
                                     10 * math.log10(255.0 ** 2 / (squared / elements)),
                                     "ssim": float(np.mean(structures)) if structures else None}
            manifest["elapsed_seconds"] = time.perf_counter() - started
            _save_manifest(root, manifest)
            log(f"[{index}/{len(paths)}] saved {filename}: MSE={mse:.6f}, PSNR={psnr:.2f}dB")
        manifest["status"] = "completed"
        return manifest
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if net is not None:
            net.skip = set()
        clear_caches()
        manifest["elapsed_seconds"] = time.perf_counter() - started
        _save_manifest(root, manifest)
