"""Offline, full-frame floating-point teacher targets for recovery distillation.

The manifest owns only files below its cache directory. Existing caches are resumed
only when their source images, weights, dimensions and controls still match.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from safetensors.torch import load_file, save_file

from . import int4, int8, quant
from .io import Controls, frame_to_tensor, load_image
from .model import NRNet
from .runtime import clear_caches, resolve_device
from .weights import Weights


_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def file_sha256(path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_target(path: Path, target: torch.Tensor) -> str:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        save_file({"target": target.contiguous()}, str(temporary))
        digest = file_sha256(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _is_hash(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _target_path(root: Path, entry: dict) -> Path:
    relative = entry.get("target")
    if not isinstance(relative, str):
        raise ValueError("cache target path is missing")
    candidate = Path(relative)
    if candidate.is_absolute() or len(candidate.parts) != 2 or candidate.parts[0] != "targets" \
            or candidate.suffix != ".safetensors" or ".." in candidate.parts:
        raise ValueError(f"unsafe cache target path: {relative}")
    path = root / candidate
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"cache target escapes the cache directory: {relative}")
    return path


def _read_target(path: Path, width: int, height: int) -> torch.Tensor:
    tensors = load_file(str(path), device="cpu")
    target = tensors.get("target")
    if set(tensors) != {"target"} or not isinstance(target, torch.Tensor) \
            or target.dtype != torch.float32 or tuple(target.shape) != (1, 3, height, width):
        raise ValueError(f"invalid float32 teacher target shape or dtype: {path}")
    if not torch.isfinite(target).all().item() or not ((target >= 0) & (target <= 1)).all().item():
        raise ValueError(f"teacher target must be finite and within [0, 1]: {path}")
    return target


def _read_manifest(root: Path) -> dict:
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int \
            or manifest["schema_version"] != 1 \
            or type(manifest.get("complete")) is not bool:
        raise ValueError("unsupported or malformed teacher cache manifest")
    if any(type(manifest.get(key)) is not int or manifest[key] < 1 for key in ("width", "height")):
        raise ValueError("cache dimensions must be positive integers")
    if not _is_hash(manifest.get("weights_sha256")) \
            or not isinstance(manifest.get("weights_path"), str) \
            or not Path(manifest["weights_path"]).is_absolute():
        raise ValueError("invalid source weights metadata")
    if manifest.get("controls") != asdict(Controls()):
        raise ValueError("teacher cache controls do not match the fixed defaults")
    entries = manifest.get("images")
    if not isinstance(entries, list) or not entries:
        raise ValueError("teacher cache contains no images")
    names, inputs, targets = set(), set(), set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) \
                or not entry["name"] or Path(entry["name"]).name != entry["name"]:
            raise ValueError("invalid cache image name")
        source = entry.get("input_path")
        if not isinstance(source, str) or not Path(source).is_absolute() \
                or not _is_hash(entry.get("input_sha256")) or type(entry.get("ready")) is not bool:
            raise ValueError("invalid cache source metadata")
        if not Path(source).is_file():
            raise FileNotFoundError(f"cached input is missing: {source}")
        target = _target_path(root, entry)
        if source in inputs or entry["name"] in names or str(target) in targets:
            raise ValueError("duplicate teacher cache image or target")
        names.add(entry["name"])
        inputs.add(source)
        targets.add(str(target))
        if entry["ready"] and not _is_hash(entry.get("target_sha256")):
            raise ValueError("ready target has no valid content hash")
        if manifest["complete"] and not entry["ready"]:
            raise ValueError("complete cache contains unfinished targets")
    protected = {Path(entry["input_path"]).resolve() for entry in entries}
    protected.add(Path(manifest["weights_path"]).resolve())
    if any(_target_path(root, entry).resolve() in protected for entry in entries):
        raise ValueError("cache target would overwrite source input or weights")
    return manifest


def _verify_ready(root: Path, manifest: dict) -> None:
    for entry in manifest["images"]:
        if entry["ready"]:
            path = _target_path(root, entry)
            if file_sha256(path) != entry["target_sha256"]:
                raise ValueError(f"teacher target content changed: {path}")
            _read_target(path, manifest["width"], manifest["height"])


class CachedFrames(torch.utils.data.Dataset):
    """One original uint8 frame and one CPU float32 [1, 3, H, W] target per item."""

    def __init__(self, cache_dir, weights_hash: str | None = None):
        self.root = Path(cache_dir).expanduser().resolve()
        self.manifest = _read_manifest(self.root)
        if not self.manifest["complete"]:
            raise ValueError("teacher cache is incomplete; resume prepare before training")
        if weights_hash is not None and self.manifest["weights_sha256"] != weights_hash:
            raise ValueError("teacher cache was generated with different weights")
        self.width, self.height = self.manifest["width"], self.manifest["height"]
        self.entries = self.manifest["images"]
        _verify_ready(self.root, self.manifest)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        if file_sha256(entry["input_path"]) != entry["input_sha256"]:
            raise ValueError(f"cached input content changed: {entry['input_path']}")
        frame = load_image(entry["input_path"])
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"cached input dimensions changed: {entry['input_path']}")
        path = _target_path(self.root, entry)
        if file_sha256(path) != entry["target_sha256"]:
            raise ValueError(f"teacher target content changed: {path}")
        return {"input": frame, "target": _read_target(path, self.width, self.height), "name": entry["name"]}


def prepare_cache(input_path, out_dir, weights_path, device="auto", log=print) -> dict:
    """Generate or resume floating-point targets with the unpruned frozen teacher.

    Source validation and compatibility checks happen before constructing the GPU
    model. Each completed target and manifest update is written atomically, so an
    interrupted invocation can be resumed with the same arguments.
    """
    if int4.active() or int8.sites or quant.skip_blocks:
        raise ValueError("teacher preparation requires the reference FP8 model; disable INT4, INT8, "
                         "and FP8-skip experiments")
    source = Path(input_path).expanduser().resolve()
    weights_path = Path(weights_path).expanduser().resolve()
    root = Path(out_dir).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights file is missing: {weights_path}")
    if source.is_dir():
        paths = [path.resolve() for path in sorted(source.iterdir(), key=lambda path: path.name)
                 if path.is_file() and path.suffix.lower() in _EXTENSIONS]
    elif source.is_file():
        paths = [source]
    else:
        raise FileNotFoundError(f"input path is missing: {source}")
    if not paths:
        raise ValueError(f"no supported images found in {source}")
    entries = []
    dimensions = None
    seen_paths = set()
    seen_names = set()
    for index, path in enumerate(paths):
        if path in seen_paths or path.name in seen_names:
            raise ValueError("teacher cache source images must have distinct paths and filenames")
        seen_paths.add(path)
        seen_names.add(path.name)
        frame = load_image(str(path))
        shape = frame.shape[:2]
        if dimensions is None:
            dimensions = shape
        if shape != dimensions:
            raise ValueError("all teacher cache images must have the same dimensions")
        entries.append({"name": path.name, "input_path": str(path), "input_sha256": file_sha256(path),
                        "target": f"targets/{index:06d}.safetensors", "ready": False})
    height, width = dimensions
    expected = {"schema_version": 1, "complete": False, "width": width, "height": height,
                "weights_sha256": file_sha256(weights_path), "weights_path": str(weights_path),
                "controls": asdict(Controls()), "images": entries}
    if root.exists() and not root.is_dir():
        raise ValueError(f"cache output is not a directory: {root}")
    if (root / "manifest.json").exists():
        manifest = _read_manifest(root)
        keys = ("schema_version", "width", "height", "weights_sha256", "controls")
        if any(manifest[key] != expected[key] for key in keys):
            raise ValueError("existing teacher cache has a different weights/controls/dimensions fingerprint")
        fields = ("name", "input_path", "input_sha256", "target")
        if len(manifest["images"]) != len(entries) or any(
            any(old[key] != new[key] for key in fields) for old, new in zip(manifest["images"], entries)
        ):
            raise ValueError("existing teacher cache inputs have changed")
        _verify_ready(root, manifest)
        if manifest["complete"]:
            log(f"Teacher cache is already complete: {root} ({len(entries)} frames)")
            return manifest
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"refusing to overwrite nonempty directory without a cache manifest: {root}")
        manifest = expected
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(root / "manifest.json", manifest)
    if all(entry["ready"] for entry in manifest["images"]):
        manifest["complete"] = True
        _atomic_json(root / "manifest.json", manifest)
        return manifest
    (root / "targets").mkdir(exist_ok=True)
    # A resume must never follow a target-directory symlink outside the cache.
    for entry in manifest["images"]:
        _target_path(root, entry)
    log(f"Preparing teacher targets: {len(entries)} frames, {width}x{height}, cache={root}")
    int4.disable()
    resolved_device = resolve_device(device)
    net = NRNet(Weights.load(str(weights_path)), width, height).to(resolved_device).eval()
    net.skip = set()
    try:
        with torch.inference_mode():
            for index, entry in enumerate(manifest["images"]):
                if entry["ready"]:
                    log(f"Teacher [{index + 1}/{len(entries)}] reusing {entry['name']}")
                    continue
                if file_sha256(entry["input_path"]) != entry["input_sha256"]:
                    raise ValueError(f"input changed during teacher preparation: {entry['input_path']}")
                frame = load_image(entry["input_path"])
                if frame.shape != (height, width, 3):
                    raise ValueError(f"input dimensions changed during teacher preparation: {entry['input_path']}")
                outputs = net.run(frame_to_tensor(frame, resolved_device), Controls(), capture=False)
                output = outputs[net.plan[-1].index]
                if isinstance(output, tuple):
                    output = output[0]
                target = output[:, :, :height, :width].detach().to(device="cpu", dtype=torch.float32).contiguous()
                if tuple(target.shape) != (1, 3, height, width) or not torch.isfinite(target).all().item() \
                        or not ((target >= 0) & (target <= 1)).all().item():
                    raise ValueError(f"invalid floating-point teacher output: {entry['name']}")
                entry["target_sha256"] = _atomic_target(_target_path(root, entry), target)
                entry["ready"] = True
                _atomic_json(root / "manifest.json", manifest)
                log(f"Teacher [{index + 1}/{len(entries)}] saved {entry['name']}")
                del target, output, outputs
        manifest["complete"] = True
        _atomic_json(root / "manifest.json", manifest)
        return manifest
    finally:
        del net
        clear_caches()
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
