"""Offline, full-frame floating-point teacher targets for recovery distillation.

The manifest owns only files below its cache directory. Existing caches are resumed
only when their source images, weights, dimensions and controls still match.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

import torch
from safetensors.torch import load_file, save_file

from . import int4, int8, quant
from .distill_losses import FEATURE_BLOCKS
from .io import Controls, frame_to_tensor, load_image
from .model import NRNet
from .plan import build_plan
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


def _atomic_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> str:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        save_file({key: value.contiguous() for key, value in tensors.items()}, str(temporary))
        digest = file_sha256(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_target(path: Path, target: torch.Tensor) -> str:
    return _atomic_tensors(path, {"target": target})


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
                or not entry["name"]:
            raise ValueError("invalid cache image name")
        name = PurePosixPath(entry["name"])
        if name.is_absolute() or ".." in name.parts or not name.parts or name.as_posix() != entry["name"]:
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


class FeatureCache:
    """Additive, mask-independent teacher features bound to an RGB cache.

    The immutable identity permits joint RGB/feature preparation to resume while
    the RGB manifest is partial. A complete sidecar also binds its exact hash.
    """

    def __init__(self, root: Path, rgb: dict, *, prepare=False):
        self.root = root
        self.directory = root / "features"
        self.path = self.directory / "manifest.json"
        if self.directory.is_symlink() or self.path.is_symlink():
            raise ValueError("feature cache paths must not be symlinks")
        identity = {key: rgb[key] for key in ("weights_sha256", "width", "height", "controls")}
        identity["images"] = [{key: entry[key] for key in ("name", "input_path", "input_sha256", "target")}
                              for entry in rgb["images"]]
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.shapes = {str(s.block): [1, s.c_out, s.out_w, s.out_h]
                       for s in build_plan(rgb["width"], rgb["height"]) if s.block in FEATURE_BLOCKS}
        expected = {"schema_version": 1, "rgb_identity_sha256": fingerprint,
                    "blocks": list(FEATURE_BLOCKS), "shapes": self.shapes, "dtype": "float16"}
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
            m = self.manifest
            if not isinstance(m, dict) or any(m.get(k) != v for k, v in expected.items()):
                raise ValueError("feature cache has incompatible taps, shapes or RGB fingerprint")
            if type(m.get("complete")) is not bool or not isinstance(m.get("images"), list) \
                    or len(m["images"]) != len(rgb["images"]):
                raise ValueError("malformed feature cache manifest")
            if m["complete"] and (not rgb["complete"] or
                                  m.get("rgb_manifest_sha256") != file_sha256(root / "manifest.json")):
                raise ValueError("feature cache RGB manifest fingerprint changed")
            for index, entry in enumerate(m["images"]):
                if not isinstance(entry, dict) or type(entry.get("ready")) is not bool \
                        or (m["complete"] and not entry["ready"]):
                    raise ValueError("malformed feature cache entry")
                if entry["ready"]:
                    self.read(index)
        elif prepare:
            if self.directory.exists() and any(self.directory.iterdir()):
                raise ValueError("refusing to overwrite nonempty feature directory without a manifest")
            self.directory.mkdir(exist_ok=True)
            self.manifest = {**expected, "complete": False,
                             "images": [{"ready": False} for _ in rgb["images"]]}
            _atomic_json(self.path, self.manifest)
        else:
            raise ValueError("teacher features are missing; rerun prepare for both caches, or use --feature-weight 0")
        if not prepare and not self.manifest["complete"]:
            raise ValueError("teacher features are incomplete; rerun prepare for both caches")

    def tensor_path(self, index):
        path = self.directory / f"{index:06d}.safetensors"
        if self.directory.is_symlink() or path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("unsafe feature cache path")
        return path

    def validate_tensors(self, tensors):
        if set(tensors) != set(self.shapes):
            raise ValueError("teacher feature blocks do not match the cache")
        for key, value in tensors.items():
            if value.dtype != torch.float16 or list(value.shape) != self.shapes[key] \
                    or not torch.isfinite(value).all().item():
                raise ValueError(f"invalid teacher feature shape, dtype or values at block {key}")

    def read(self, index):
        path = self.tensor_path(index)
        entry = self.manifest["images"][index]
        if not entry["ready"] or not _is_hash(entry.get("sha256")) or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"teacher feature content changed or incomplete: {path}")
        tensors = load_file(str(path), device="cpu")
        self.validate_tensors(tensors)
        return {int(key): value for key, value in tensors.items()}

    def write(self, index, features):
        tensors = {str(key): value.detach().to(device="cpu", dtype=torch.float16).contiguous()
                   for key, value in features.items()}
        self.validate_tensors(tensors)
        digest = _atomic_tensors(self.tensor_path(index), tensors)
        self.manifest["images"][index] = {"ready": True, "sha256": digest}
        _atomic_json(self.path, self.manifest)

    def finish(self):
        if not all(entry["ready"] for entry in self.manifest["images"]):
            raise ValueError("cannot finish an incomplete feature cache")
        self.manifest.update(complete=True, rgb_manifest_sha256=file_sha256(self.root / "manifest.json"))
        _atomic_json(self.path, self.manifest)


class CachedFrames(torch.utils.data.Dataset):
    """One original uint8 frame and one CPU float32 [1, 3, H, W] target per item."""

    def __init__(self, cache_dir, weights_hash: str | None = None, *, features=False):
        self.root = Path(cache_dir).expanduser().resolve()
        self.manifest = _read_manifest(self.root)
        if not self.manifest["complete"]:
            raise ValueError("teacher cache is incomplete; resume prepare before training")
        if weights_hash is not None and self.manifest["weights_sha256"] != weights_hash:
            raise ValueError("teacher cache was generated with different weights")
        self.width, self.height = self.manifest["width"], self.manifest["height"]
        self.entries = self.manifest["images"]
        _verify_ready(self.root, self.manifest)
        self.features = FeatureCache(self.root, self.manifest) if features else None

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
        item = {"input": frame, "target": _read_target(path, self.width, self.height), "name": entry["name"]}
        if self.features is not None:
            item["features"] = self.features.read(index)
        return item


def prepare_cache(input_path, out_dir, weights_path, device="auto", log=print, *, features=True) -> dict:
    """Generate or resume floating-point targets with the unpruned frozen teacher.

    Source validation and compatibility checks happen before constructing the GPU
    model. Input directories are searched recursively; names preserve relative
    scene paths. Each completed target and manifest update is written atomically, so an
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
        paths = [path for path in sorted(source.rglob("*"), key=lambda path: path.relative_to(source).as_posix())
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
        name = path.relative_to(source).as_posix() if source.is_dir() else path.name
        path = path.resolve()
        if path in seen_paths or name in seen_names:
            raise ValueError("teacher cache source images must have distinct paths and relative names")
        seen_paths.add(path)
        seen_names.add(name)
        frame = load_image(str(path))
        shape = frame.shape[:2]
        if dimensions is None:
            dimensions = shape
        if shape != dimensions:
            raise ValueError("all teacher cache images must have the same dimensions")
        entries.append({"name": name, "input_path": str(path), "input_sha256": file_sha256(path),
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
        if manifest["complete"] and not features:
            log(f"Teacher cache is already complete: {root} ({len(entries)} frames)")
            return manifest
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"refusing to overwrite nonempty directory without a cache manifest: {root}")
        manifest = expected
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(root / "manifest.json", manifest)
    feature_cache = FeatureCache(root, manifest, prepare=True) if features else None
    if not manifest["complete"] and all(entry["ready"] for entry in manifest["images"]):
        manifest["complete"] = True
        _atomic_json(root / "manifest.json", manifest)
    if manifest["complete"] and (feature_cache is None or feature_cache.manifest["complete"]):
        log(f"Teacher cache is already complete: {root} ({len(entries)} frames)")
        return manifest
    if manifest["complete"] and all(e["ready"] for e in feature_cache.manifest["images"]):
        feature_cache.finish()
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
                need_features = feature_cache is not None and not feature_cache.manifest["images"][index]["ready"]
                if entry["ready"] and not need_features:
                    log(f"Teacher [{index + 1}/{len(entries)}] reusing {entry['name']}")
                    continue
                if file_sha256(entry["input_path"]) != entry["input_sha256"]:
                    raise ValueError(f"input changed during teacher preparation: {entry['input_path']}")
                frame = load_image(entry["input_path"])
                if frame.shape != (height, width, 3):
                    raise ValueError(f"input dimensions changed during teacher preparation: {entry['input_path']}")
                captured = {}

                def capture_feature(spec, value):
                    if spec.block in FEATURE_BLOCKS:
                        value = value[0] if isinstance(value, tuple) else value
                        captured[spec.block] = value.detach().to(device="cpu", dtype=torch.float16)

                options = {"on_layer": capture_feature} if need_features else {}
                outputs = net.run(frame_to_tensor(frame, resolved_device), Controls(), capture=False, **options)
                output = outputs[net.plan[-1].index]
                if isinstance(output, tuple):
                    output = output[0]
                target = output[:, :, :height, :width].detach().to(device="cpu", dtype=torch.float32).contiguous()
                if tuple(target.shape) != (1, 3, height, width) or not torch.isfinite(target).all().item() \
                        or not ((target >= 0) & (target <= 1)).all().item():
                    raise ValueError(f"invalid floating-point teacher output: {entry['name']}")
                if not entry["ready"]:
                    entry["target_sha256"] = _atomic_target(_target_path(root, entry), target)
                    entry["ready"] = True
                    _atomic_json(root / "manifest.json", manifest)
                if need_features:
                    feature_cache.write(index, captured)
                log(f"Teacher [{index + 1}/{len(entries)}] saved {entry['name']}")
                del target, output, outputs, captured
        if not manifest["complete"]:
            manifest["complete"] = True
            _atomic_json(root / "manifest.json", manifest)
        if feature_cache is not None:
            feature_cache.finish()
        return manifest
    finally:
        del net
        clear_caches()
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
