"""Validation deduplication compares bytes and never removes training files."""
from pathlib import Path

import pytest

from scripts import remove_val_duplicates as script


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture
def dataset(tmp_path):
    train, val = tmp_path / "train", tmp_path / "val"
    write(train, "scene1/frame.png", b"shared image bytes")
    write(train, "scene2/frame.png", b"training only")
    write(val, "scene3/different-name.PNG", b"shared image bytes")
    write(val, "scene4/frame.jpg", b"shared image bytes")
    write(val, "scene5/frame.png", b"validation only")
    write(val, "scene6/frame.png", b"validation only")
    write(val, "notes.txt", b"shared image bytes")
    return train, val


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_removes_only_cross_split_duplicates_and_is_repeatable(dataset):
    train, val = dataset
    original_train = snapshot(train)
    result = script.remove_validation_duplicates(train, val, log=lambda _: None)
    assert result == {"training_images": 2, "validation_images": 4, "matches": 2, "removed": 2, "dry_run": False}
    assert snapshot(train) == original_train
    assert set(snapshot(val)) == {Path("scene5/frame.png"), Path("scene6/frame.png"), Path("notes.txt")}
    repeated = script.remove_validation_duplicates(train, val, log=lambda _: None)
    assert repeated["matches"] == repeated["removed"] == 0


def test_dry_run_reports_hashes_and_paths_without_changes(dataset, capsys):
    train, val = dataset
    before = snapshot(val)
    assert script.main(["--train-dir", str(train), "--val-dir", str(val), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert script.file_sha256(train / "scene1/frame.png") in output
    assert "scene3/different-name.PNG" in output and "Would remove:" in output
    assert "removed 0" in output and snapshot(val) == before


def test_default_cli_removes_duplicates_and_mentions_cache_rebuild(dataset, capsys):
    train, val = dataset
    assert script.main(["--train-dir", str(train), "--val-dir", str(val)]) == 0
    output = capsys.readouterr().out
    assert "removed 2" in output and "new output directory" in output
    assert (train / "scene1/frame.png").exists()
    assert not (val / "scene3/different-name.PNG").exists()


@pytest.mark.parametrize("relationship", ["same", "val_inside_train", "train_inside_val"])
def test_overlapping_directories_rejected_without_removals(tmp_path, relationship):
    outer, inner = tmp_path / "data", tmp_path / "data/nested"
    write(outer, "a.png", b"same")
    write(inner, "b.png", b"same")
    train, val = {"same": (outer, outer), "val_inside_train": (outer, inner),
                  "train_inside_val": (inner, outer)}[relationship]
    before = snapshot(outer)
    with pytest.raises(ValueError, match="separate and not nested"):
        script.remove_validation_duplicates(train, val)
    assert snapshot(outer) == before


def test_symlinks_are_skipped_and_link_targets_preserved(dataset, tmp_path):
    train, val = dataset
    external = write(tmp_path / "outside", "external.png", b"shared image bytes")
    (val / "linked.png").symlink_to(external)
    (val / "linked-scene").symlink_to(external.parent, target_is_directory=True)
    logs = []
    result = script.remove_validation_duplicates(train, val, log=logs.append)
    assert result["removed"] == 2 and external.read_bytes() == b"shared image bytes"
    assert (val / "linked.png").is_symlink() and (val / "linked-scene").is_symlink()
    assert any("Skipping symlink" in line for line in logs)


@pytest.mark.parametrize("change", ["training", "validation"])
def test_changed_files_are_not_removed(dataset, change):
    train, val = dataset
    changing = train / "scene1/frame.png" if change == "training" else val / "scene3/different-name.PNG"

    def log(message):
        if message.startswith("SHA256:"):
            changing.write_bytes(b"changed after scan")

    with pytest.raises(ValueError, match="contents changed"):
        script.remove_validation_duplicates(train, val, log=log)
    assert (val / "scene3/different-name.PNG").exists()
    assert (val / "scene4/frame.jpg").exists()


def test_missing_empty_training_and_empty_validation(tmp_path, capsys):
    train, val = tmp_path / "train", tmp_path / "val"
    assert script.main(["--train-dir", str(train), "--val-dir", str(val)]) == 1
    assert "does not exist" in capsys.readouterr().err
    train.mkdir()
    val.mkdir()
    with pytest.raises(ValueError, match="no supported training images"):
        script.remove_validation_duplicates(train, val)
    write(train, "frame.png", b"image")
    result = script.remove_validation_duplicates(train, val, log=lambda _: None)
    assert result["validation_images"] == result["removed"] == 0
