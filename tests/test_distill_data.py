"""Teacher-cache tests use a tiny fake network, never the production GPU model."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from safetensors.torch import load_file, save_file

from dlssnr import distill_data as data
from dlssnr.io import Controls


def image(path: Path, value=50, shape=(3, 4, 3)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(shape, value, dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"test teacher weights")
    source = tmp_path / "images"
    image(source / "b.PNG", value=80)
    image(source / "a.png")
    calls = []
    instances = []

    class FakeNet:
        def __init__(self, weights, width, height):
            self.plan = [SimpleNamespace(index=151)]
            self.width, self.height = width, height
            self.skip = {1, 2}
            instances.append(self)

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.training = False
            return self

        def run(self, tensor, controls, capture=False, *, on_layer=None):
            assert self.skip == set()
            assert not self.training
            assert torch.is_inference_mode_enabled()
            assert controls == Controls()
            assert not capture
            calls.append(tensor.clone())
            if on_layer is not None:
                for spec in data.build_plan(self.width, self.height):
                    if spec.block in data.FEATURE_BLOCKS:
                        value = torch.full((1, spec.c_out, spec.out_w, spec.out_h), 0.125)
                        on_layer(spec, (value, value + 1) if spec.variant == "ds" else value)
            return {151: torch.full_like(tensor, 0.1234567)}

    monkeypatch.setattr(data, "NRNet", FakeNet)
    monkeypatch.setattr(data.Weights, "load", lambda path: object())
    return SimpleNamespace(root=tmp_path / "cache", weights=weights, source=source, calls=calls,
                           instances=instances, fake_net=FakeNet)


def prepare(setup, *, features=False):
    return data.prepare_cache(setup.source, setup.root, setup.weights, device="cpu", log=lambda _: None,
                              features=features)


def write_manifest(setup, manifest):
    (setup.root / "manifest.json").write_text(json.dumps(manifest))


def test_hash_streams_file(tmp_path):
    path = tmp_path / "large"
    content = b"a" * (2 * 1024 * 1024 + 3)
    path.write_bytes(content)
    assert data.file_sha256(path) == hashlib.sha256(content).hexdigest()


def test_prepare_saves_float_outputs_and_complete_metadata(setup):
    manifest = prepare(setup)
    assert manifest["complete"] and manifest["schema_version"] == 1
    assert manifest["width"] == 4 and manifest["height"] == 3
    assert manifest["weights_sha256"] == data.file_sha256(setup.weights)
    assert manifest["weights_path"] == str(setup.weights)
    assert manifest["controls"] == asdict(Controls())
    assert [entry["name"] for entry in manifest["images"]] == ["a.png", "b.PNG"]
    assert len(setup.calls) == 2
    # Teacher receives padded whole frames; target stores only the original extent.
    assert setup.calls[0].shape[-2] >= 3 and setup.calls[0].shape[-1] >= 4
    target = load_file(str(setup.root / manifest["images"][0]["target"]))["target"]
    assert target.dtype == torch.float32 and target.shape == (1, 3, 3, 4)
    assert target.mean().item() == pytest.approx(0.1234567)
    assert target.mean().item() != pytest.approx(round(0.1234567 * 255) / 255)
    assert not list(setup.root.rglob("*.tmp"))


def test_dataset_returns_original_image_and_float_target(setup):
    prepare(setup)
    dataset = data.CachedFrames(setup.root, weights_hash=data.file_sha256(setup.weights))
    assert len(dataset) == 2 and (dataset.width, dataset.height) == (4, 3)
    assert dataset.root == setup.root and dataset.entries == dataset.manifest["images"]
    item = dataset[0]
    assert item["name"] == "a.png"
    assert item["input"].dtype == np.uint8 and item["input"].shape == (3, 4, 3)
    assert (item["input"] == 50).all()
    assert item["target"].dtype == torch.float32 and item["target"].device.type == "cpu"
    assert not item["target"].requires_grad


def test_complete_cache_is_reused_without_teacher(setup):
    first = prepare(setup)
    assert prepare(setup) == first
    assert len(setup.instances) == 1 and len(setup.calls) == 2


def test_prepare_accepts_single_image_and_recurses_into_scene_directories(setup):
    image(setup.source / "scene2" / "frame.png", value=90)
    image(setup.source / "scene1" / "frame.png", value=60)
    image(setup.source / "scene1" / "sequence" / "frame.png", value=70)
    (setup.source / "readme.txt").write_text("skip")
    (setup.source / "scene1" / "readme.txt").write_text("skip")
    manifest = prepare(setup, features=True)
    names = ["a.png", "b.PNG", "scene1/frame.png", "scene1/sequence/frame.png", "scene2/frame.png"]
    assert [entry["name"] for entry in manifest["images"]] == names
    assert len({entry["target"] for entry in manifest["images"]}) == 5
    dataset = data.CachedFrames(setup.root, features=True)
    assert [dataset[i]["name"] for i in range(len(dataset))] == names
    assert [int(dataset[i]["input"][0, 0, 0]) for i in range(len(dataset))] == [50, 80, 60, 70, 90]
    assert prepare(setup, features=True) == manifest
    assert len(setup.calls) == 5
    other = setup.root.parent / "single"
    manifest = data.prepare_cache(setup.source / "scene1" / "frame.png", other, setup.weights, "cpu", log=lambda _: None)
    assert len(manifest["images"]) == 1
    assert manifest["images"][0]["name"] == "frame.png"


@pytest.mark.parametrize("name", ["", ".", "../frame.png", "/scene/frame.png", "scene/../frame.png",
                                  "scene//frame.png", "./frame.png"])
def test_manifest_rejects_noncanonical_or_escaping_image_names(setup, name):
    manifest = prepare(setup)
    manifest["images"][0]["name"] = name
    write_manifest(setup, manifest)
    with pytest.raises(ValueError, match="invalid cache image name"):
        data.CachedFrames(setup.root)


@pytest.mark.parametrize("nested", [False, True])
def test_interruption_preserves_partial_manifest_and_resume_reuses_completed(setup, monkeypatch, nested):
    if nested:
        for scene, filename in (("scene1", "a.png"), ("scene2", "b.PNG")):
            (setup.source / scene).mkdir()
            (setup.source / filename).rename(setup.source / scene / "frame.png")
    original = setup.fake_net.run
    runs = 0

    def interrupt(self, *args, **kwargs):
        nonlocal runs
        runs += 1
        if runs == 2:
            raise KeyboardInterrupt
        return original(self, *args, **kwargs)

    monkeypatch.setattr(setup.fake_net, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        prepare(setup)
    partial = json.loads((setup.root / "manifest.json").read_text())
    assert not partial["complete"]
    assert [entry["ready"] for entry in partial["images"]] == [True, False]
    with pytest.raises(ValueError, match="incomplete"):
        data.CachedFrames(setup.root)
    monkeypatch.setattr(setup.fake_net, "run", original)
    assert prepare(setup)["complete"]
    assert len(setup.calls) == 2


@pytest.mark.parametrize("change", ["weights", "image", "dimensions", "controls"])
def test_prepare_rejects_stale_fingerprint_before_gpu(setup, change):
    manifest = prepare(setup)
    if change == "weights":
        setup.weights.write_bytes(b"changed weights")
    elif change == "image":
        image(setup.source / "a.png", value=51)
    elif change == "dimensions":
        for entry in manifest["images"]:
            image(Path(entry["input_path"]), shape=(4, 4, 3))
    else:
        manifest["controls"]["seed"] = 1
        write_manifest(setup, manifest)
    with pytest.raises(ValueError):
        prepare(setup)
    assert len(setup.instances) == 1


def test_dimension_mismatch_rejected_before_gpu_and_writes(setup):
    image(setup.source / "different.png", shape=(4, 4, 3))
    with pytest.raises(ValueError, match="same dimensions"):
        prepare(setup)
    assert not setup.instances and not setup.root.exists()


def test_rejects_noncache_output_without_overwriting(setup):
    setup.root.mkdir()
    protected = setup.root / "precious.txt"
    protected.write_text("keep")
    with pytest.raises(ValueError, match="nonempty"):
        prepare(setup)
    assert protected.read_text() == "keep" and not setup.instances


def test_missing_inputs_and_weights_are_rejected(setup):
    with pytest.raises(FileNotFoundError):
        data.prepare_cache(setup.source / "missing", setup.root, setup.weights)
    with pytest.raises(FileNotFoundError):
        data.prepare_cache(setup.source, setup.root, setup.weights.parent / "missing")
    empty = setup.source.parent / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no supported images"):
        data.prepare_cache(empty, setup.root, setup.weights)
    with pytest.raises(FileNotFoundError):
        data.CachedFrames(setup.root)


def test_dataset_rejects_wrong_weights_and_modified_inputs(setup):
    prepare(setup)
    with pytest.raises(ValueError, match="different weights"):
        data.CachedFrames(setup.root, weights_hash="f" * 64)
    dataset = data.CachedFrames(setup.root)
    image(setup.source / "a.png", value=1)
    with pytest.raises(ValueError, match="input content changed"):
        dataset[0]


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "range", "key"])
def test_dataset_rejects_invalid_targets_even_with_matching_hash(setup, kind):
    manifest = prepare(setup)
    entry = manifest["images"][0]
    target_path = setup.root / entry["target"]
    target = torch.zeros((1, 3, 3, 4))
    if kind == "shape":
        target = target[:, :, :1]
    elif kind == "dtype":
        target = target.double()
    elif kind == "nan":
        target.fill_(float("nan"))
    elif kind == "range":
        target.fill_(1.1)
    save_file({"bad" if kind == "key" else "target": target.contiguous()}, str(target_path))
    entry["target_sha256"] = data.file_sha256(target_path)
    write_manifest(setup, manifest)
    with pytest.raises(ValueError):
        data.CachedFrames(setup.root)


def test_target_hash_checked_at_load_and_at_item_access(setup):
    manifest = prepare(setup)
    dataset = data.CachedFrames(setup.root)
    path = setup.root / manifest["images"][0]["target"]
    save_file({"target": torch.zeros((1, 3, 3, 4))}, str(path))
    with pytest.raises(ValueError, match="target content changed"):
        data.CachedFrames(setup.root)
    with pytest.raises(ValueError, match="target content changed"):
        dataset[0]


@pytest.mark.parametrize("target", ["../outside.safetensors", "/tmp/target.safetensors", "targets/../a.safetensors"])
def test_target_path_cannot_escape_cache(setup, target):
    manifest = prepare(setup)
    manifest["images"][0]["target"] = target
    write_manifest(setup, manifest)
    with pytest.raises(ValueError, match="unsafe cache target path"):
        data.CachedFrames(setup.root)


def test_invalid_teacher_target_leaves_incomplete_cache(setup, monkeypatch):
    def invalid(self, tensor, controls, capture=False):
        return {151: torch.full_like(tensor, float("nan"))}

    monkeypatch.setattr(setup.fake_net, "run", invalid)
    with pytest.raises(ValueError, match="invalid floating-point"):
        prepare(setup)
    manifest = json.loads((setup.root / "manifest.json").read_text())
    assert not manifest["complete"] and not any(entry["ready"] for entry in manifest["images"])
    assert not list(setup.root.rglob("*.safetensors"))


@pytest.mark.parametrize("experiment", ["int4", "int8", "fp8-skip"])
def test_prepare_rejects_nonreference_experiments_before_writes(setup, monkeypatch, experiment):
    if experiment == "int4":
        monkeypatch.setattr(data.int4, "active", lambda: True)
    elif experiment == "int8":
        monkeypatch.setattr(data.int8, "sites", {1})
    else:
        monkeypatch.setattr(data.quant, "skip_blocks", {1})
    with pytest.raises(ValueError, match="requires the reference FP8 model"):
        prepare(setup)
    assert not setup.root.exists() and not setup.instances


def test_feature_upgrade_preserves_rgb_and_complete_cache_reuses_without_teacher(setup):
    rgb = prepare(setup)
    paths = [setup.root / "manifest.json"] + [setup.root / e["target"] for e in rgb["images"]]
    original = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
    prepare(setup, features=True)
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths] == original
    dataset = data.CachedFrames(setup.root, features=True)
    assert set(dataset[0]["features"]) == set(data.FEATURE_BLOCKS)
    assert all(v.dtype == torch.float16 and (v == 0.125).all() for v in dataset[0]["features"].values())
    calls = len(setup.calls)
    instances = len(setup.instances)
    sidecar = (setup.root / "features/manifest.json").read_bytes()
    prepare(setup, features=True)
    assert len(setup.calls) == calls and len(setup.instances) == instances
    assert (setup.root / "features/manifest.json").read_bytes() == sidecar
    assert not list(setup.root.rglob("*.tmp"))


def test_fresh_features_default_uses_one_teacher_pass_per_image(setup):
    data.prepare_cache(setup.source, setup.root, setup.weights, "cpu", log=lambda _: None)
    assert len(setup.calls) == 2 and len(setup.instances) == 1
    assert data.CachedFrames(setup.root, features=True).features.manifest["complete"]


def test_partial_feature_upgrade_resumes_without_replacing_rgb(setup, monkeypatch):
    prepare(setup)
    rgb_bytes = (setup.root / "manifest.json").read_bytes()
    original = setup.fake_net.run
    runs = 0

    def interrupted(self, *args, **kwargs):
        nonlocal runs
        runs += 1
        if runs == 2:
            raise KeyboardInterrupt
        return original(self, *args, **kwargs)

    monkeypatch.setattr(setup.fake_net, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        prepare(setup, features=True)
    with pytest.raises(ValueError, match="features are incomplete"):
        data.CachedFrames(setup.root, features=True)
    assert len(data.CachedFrames(setup.root)) == 2
    first_path = setup.root / "features/000000.safetensors"
    first_time = first_path.stat().st_mtime_ns
    monkeypatch.setattr(setup.fake_net, "run", original)
    prepare(setup, features=True)
    assert first_path.stat().st_mtime_ns == first_time
    assert (setup.root / "manifest.json").read_bytes() == rgb_bytes
    assert len(setup.calls) == 4  # Two RGB passes and two successful feature passes.


@pytest.mark.parametrize("change", ["fingerprint", "taps", "shape", "dtype", "nan", "hash"])
def test_feature_cache_rejects_stale_or_invalid_features(setup, change):
    prepare(setup, features=True)
    manifest_path = setup.root / "features/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = setup.root / "features/000000.safetensors"
    if change == "fingerprint":
        manifest["rgb_manifest_sha256"] = "f" * 64
    elif change == "taps":
        manifest["blocks"] = [1, 2, 3]
    elif change == "hash":
        manifest["images"][0]["sha256"] = "f" * 64
    else:
        tensors = load_file(str(path))
        key = next(iter(tensors))
        if change == "shape":
            tensors[key] = tensors[key][..., :1].contiguous()
        elif change == "dtype":
            tensors[key] = tensors[key].float()
        else:
            tensors[key].fill_(float("nan"))
        save_file(tensors, str(path))
        manifest["images"][0]["sha256"] = data.file_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        data.CachedFrames(setup.root, features=True)
    instances = len(setup.instances)
    with pytest.raises(ValueError):
        prepare(setup, features=True)
    assert len(setup.instances) == instances


def test_feature_hash_checked_again_on_access_and_missing_cache_is_actionable(setup):
    prepare(setup)
    with pytest.raises(ValueError, match="rerun prepare"):
        data.CachedFrames(setup.root, features=True)
    prepare(setup, features=True)
    dataset = data.CachedFrames(setup.root, features=True)
    (setup.root / "features/000000.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="feature content changed"):
        dataset[0]


def test_feature_directory_symlink_rejected_without_external_writes(setup, tmp_path):
    prepare(setup)
    outside = tmp_path / "external"
    outside.mkdir()
    (setup.root / "features").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        prepare(setup, features=True)
    assert not list(outside.iterdir())
