"""Remove validation images whose SHA-256 matches an image in training.

    python scripts/remove_val_duplicates.py --train-dir data/train --val-dir data/val --dry-run
    python scripts/remove_val_duplicates.py --train-dir data/train --val-dir data/val

Searches scene subdirectories and compares current file contents, not cache
manifests or filenames. Only matching validation files are removed. Rebuild the
validation teacher cache in a new directory afterward. Uses only the standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys


# Keep the supported input formats aligned with scripts/distill.py prepare.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(root: Path, log) -> list[Path]:
    def fail(error):
        raise error

    paths = []
    for directory, subdirs, filenames in os.walk(root, followlinks=False, onerror=fail):
        base = Path(directory)
        for name in list(subdirs):
            if (base / name).is_symlink():
                log(f"Skipping symlink directory: {base / name}")
                subdirs.remove(name)
        for name in filenames:
            path = base / name
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if path.is_symlink():
                log(f"Skipping symlink image: {path}")
            elif path.is_file():
                paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def remove_validation_duplicates(train_dir, val_dir, *, dry_run=False, log=print) -> dict:
    train_root = Path(train_dir).expanduser().resolve()
    val_root = Path(val_dir).expanduser().resolve()
    for root in (train_root, val_root):
        if not root.is_dir():
            raise ValueError(f"dataset directory does not exist: {root}")
    if train_root == val_root or train_root in val_root.parents or val_root in train_root.parents:
        raise ValueError("training and validation directories must be separate and not nested")

    train_files = image_paths(train_root, log)
    val_files = image_paths(val_root, log)
    if not train_files:
        raise ValueError(f"no supported training images found in {train_root}")
    log(f"Hashing {len(train_files)} training images and {len(val_files)} validation images")
    training_hashes = {}
    for index, path in enumerate(train_files, 1):
        training_hashes.setdefault(file_sha256(path), path)
        if index % 100 == 0:
            log(f"Training hashes: {index}/{len(train_files)}")
    matches = []
    for index, path in enumerate(val_files, 1):
        digest = file_sha256(path)
        if digest in training_hashes:
            matches.append((digest, training_hashes[digest], path))
        if index % 100 == 0:
            log(f"Validation hashes: {index}/{len(val_files)}")

    removed = 0
    for digest, training_path, validation_path in matches:
        log(f"SHA256: {digest}\n  TRAIN: {training_path}\n  VAL:   {validation_path}")
        if dry_run:
            log(f"Would remove: {validation_path}")
            continue
        # Recheck before unlinking, including path containment if a directory was
        # replaced by a symlink while a large dataset was being scanned.
        if validation_path.is_symlink() or not validation_path.resolve().is_relative_to(val_root) \
                or training_path.is_symlink() or not training_path.resolve().is_relative_to(train_root):
            raise ValueError("dataset paths changed during scanning; stopping removal")
        if file_sha256(validation_path) != digest or file_sha256(training_path) != digest:
            raise ValueError("dataset contents changed during scanning; stopping removal")
        validation_path.unlink()
        removed += 1
        log(f"Removed: {validation_path}")

    result = {"training_images": len(train_files), "validation_images": len(val_files),
              "matches": len(matches), "removed": removed, "dry_run": dry_run}
    log(f"Matched {len(matches)}/{len(val_files)} validation images; removed {removed}."
        + (" Dry run: no files changed." if dry_run else ""))
    if removed:
        log("Rebuild the validation teacher cache in a new output directory before training.")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", default="data/train", help="training image directory (default: data/train)")
    parser.add_argument("--val-dir", default="data/val", help="validation image directory (default: data/val)")
    parser.add_argument("--dry-run", action="store_true", help="list matching validation files without removing them")
    args = parser.parse_args(argv)
    try:
        remove_validation_duplicates(args.train_dir, args.val_dir, dry_run=args.dry_run,
                                     log=lambda message: print(message, flush=True))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; any reported removals have already taken effect.", file=sys.stderr)
        return 130
    except (OSError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
