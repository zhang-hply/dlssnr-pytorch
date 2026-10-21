"""Full-frame recovery training, resume checkpoints and inference-format evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file, save_file

from .distill_data import CachedFrames, file_sha256
from .distill_logging import image_callback, load_summary_writer, write_metrics, write_scalar
from .distillation import DistilledStudent, read_pruning_report, require_reference, validate_pruning
from .io import Controls, frame_to_tensor, tensor_to_frame
from .metrics import ssim
from .model import NRNet
from .plan import build_plan
from .runtime import clear_caches, resolve_device
from .weights import Weights


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    learning_rate: float = 1e-5
    min_learning_rate: float = 1e-6
    accumulation: int = 4
    grad_clip: float = 1.0
    patience: int = 5
    seed: int = 0
    checkpoint_blocks: bool = True

    def validate(self) -> None:
        if self.epochs < 1 or self.accumulation < 1 or self.patience < 1:
            raise ValueError("epochs, accumulation and patience must be positive")
        if not all(math.isfinite(value) and value > 0 for value in (
                self.learning_rate, self.min_learning_rate, self.grad_clip)):
            raise ValueError("learning rates and gradient clipping must be finite and positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("minimum learning rate must not exceed the initial learning rate")


def distillation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 plus 0.1 times the mean horizontal/vertical finite-difference L1."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have matching NCHW shapes")
    pixel = (prediction - target).abs().mean()
    horizontal = ((prediction[..., 1:] - prediction[..., :-1]) -
                  (target[..., 1:] - target[..., :-1])).abs().mean() if prediction.shape[-1] > 1 else pixel * 0
    vertical = ((prediction[..., 1:, :] - prediction[..., :-1, :]) -
                (target[..., 1:, :] - target[..., :-1, :])).abs().mean() if prediction.shape[-2] > 1 else pixel * 0
    return pixel + 0.05 * (horizontal + vertical)


def _safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    return value


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict) -> None:
    _atomic_write(path, lambda temporary: temporary.write_text(
        json.dumps(_safe_json(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"))


def _psnr(mse: float, maximum: float = 1.0) -> float:
    return math.inf if mse == 0 else 10 * math.log10(maximum * maximum / mse)


def _new_network(weights_path, width, height):
    return NRNet(Weights.load(str(weights_path)), width, height)


def _check_data(train: CachedFrames, validation: CachedFrames) -> None:
    if (train.width, train.height) != (validation.width, validation.height):
        raise ValueError("training and validation dimensions differ; no automatic resizing is performed")
    overlap = {entry["input_sha256"] for entry in train.entries} & {
        entry["input_sha256"] for entry in validation.entries}
    if overlap:
        raise ValueError("training and validation contain identical source images; split by scene first")


def evaluate_network(net, dataset: CachedFrames, device="auto", log=print, *, on_image=None) -> dict:
    """Score the ordinary inference graph against float targets and their 8-bit frames."""
    require_reference(net)
    device = resolve_device(str(device))
    squared, byte_squared, elements = 0.0, 0.0, 0
    loss_sum = 0.0
    structures, per_image = [], []
    started = time.perf_counter()
    clear_caches()
    try:
        net.to(device).eval()
        with torch.inference_mode():
            for index in range(len(dataset)):
                item = dataset[index]
                log(f"Validation {index + 1}/{len(dataset)}: {item['name']} starting")
                image_started = time.perf_counter()
                image = frame_to_tensor(item["input"], device)
                outputs = net.run(image, Controls(), capture=False)
                value = outputs[net.plan[-1].index]
                if isinstance(value, tuple):
                    value = value[0]
                prediction = value[..., :dataset.height, :dataset.width].float().cpu()
                target = item["target"]
                if prediction.shape != target.shape or not torch.isfinite(prediction).all().item():
                    raise ValueError("inference returned an invalid image")
                mse = float((prediction.double() - target.double()).square().mean())
                loss = float(distillation_loss(prediction, target))
                loss_sum += loss
                output_frame = tensor_to_frame(prediction, dataset.width, dataset.height)
                target_frame = tensor_to_frame(target, dataset.width, dataset.height)
                byte_mse = float(np.square(output_frame.astype(np.float64) - target_frame.astype(np.float64)).mean())
                structure = ssim(output_frame, target_frame) if min(dataset.width, dataset.height) > 5 else None
                if structure is not None:
                    structures.append(structure)
                squared += mse * target.numel()
                byte_squared += byte_mse * target.numel()
                elements += target.numel()
                per_image.append({"name": item["name"], "loss": loss, "mse": mse, "psnr_db": _psnr(mse),
                                  "mse_uint8": byte_mse, "psnr_uint8_db": _psnr(byte_mse, 255),
                                  "ssim": structure, "seconds": time.perf_counter() - image_started})
                log(f"Validation {index + 1}/{len(dataset)}: MSE={mse:.8g}, "
                    f"PSNR={_psnr(mse):.2f}dB, uint8 PSNR={_psnr(byte_mse, 255):.2f}dB")
                if on_image is not None:
                    on_image(index, item["name"], item["input"], target_frame, output_frame)
                del outputs, value, image
    finally:
        net.to("cpu")
        clear_caches()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"loss": loss_sum / len(dataset), "mse": squared / elements, "psnr_db": _psnr(squared / elements),
            "mse_uint8": byte_squared / elements, "psnr_uint8_db": _psnr(byte_squared / elements, 255),
            "ssim": float(np.mean(structures)) if structures else None,
            "seconds": time.perf_counter() - started, "per_image": per_image}


def _export(student, metadata: dict, destination: Path) -> None:
    state = student.export_state()
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("cannot export nonfinite weights")
    _atomic_write(destination, lambda temporary: save_file(
        state, str(temporary), metadata={"format": "dlssnr-distilled", "schema_version": "1"}))
    _write_json(destination.with_suffix(".json"), metadata)


def _validate_metadata(metadata: dict, weights_hash: str) -> None:
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("unsupported distilled model metadata")
    if metadata.get("weights_sha256") != weights_hash:
        raise ValueError("distilled model requires the original teacher weights used during training")
    if metadata.get("controls") != asdict(Controls()):
        raise ValueError("distilled model controls do not match the fixed defaults")
    if any(type(metadata.get(key)) is not int or metadata[key] < 1 for key in ("width", "height")):
        raise ValueError("invalid distilled model dimensions")
    validate_pruning(build_plan(metadata["width"], metadata["height"]), metadata.get("dropped_blocks", []))


def load_export(weights_path, checkpoint_path, device="cpu"):
    """Load materialized buffers into the unmodified NRNet inference entry point.

    Source extraction weights supply the network skeleton; the safetensors state
    replaces its buffers. The source identity and fixed pruning mask are checked.
    """
    require_reference()
    source, checkpoint_path = Path(weights_path).expanduser().resolve(), Path(checkpoint_path).expanduser().resolve()
    with checkpoint_path.with_suffix(".json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    _validate_metadata(metadata, file_sha256(source))
    net = _new_network(source, metadata["width"], metadata["height"])
    net.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=True)
    net.skip = set(validate_pruning(net.plan, metadata["dropped_blocks"])["layer_indices"])
    return net.to(resolve_device(str(device))).eval(), metadata


def evaluate_export(weights_path, checkpoint_path, cache_dir, out_path=None, device="auto", log=print,
                    *, on_image=None, images_out=None) -> dict:
    """Evaluate an export, optionally saving full-resolution input/teacher/student PNGs.

    images_out must be a new or empty directory. The image callback still receives
    every frame when disk comparisons are enabled; neither output needs extra inference.
    """
    # Validate all artifacts before moving the model to the device.
    source = Path(weights_path).expanduser().resolve()
    dataset = CachedFrames(cache_dir, weights_hash=file_sha256(source))
    destination = None
    if out_path is not None:
        destination = Path(out_path).expanduser().resolve()
        protected = {source, Path(checkpoint_path).expanduser().resolve(),
                     Path(checkpoint_path).expanduser().resolve().with_suffix(".json"), dataset.root / "manifest.json"}
        protected.update(Path(entry["input_path"]).resolve() for entry in dataset.entries)
        protected.update((dataset.root / entry["target"]).resolve() for entry in dataset.entries)
        if destination in protected:
            raise ValueError("evaluation report would overwrite an input artifact")
    images_root, comparisons = None, None
    if images_out is not None:
        images_root = Path(images_out).expanduser().resolve()
        if images_root.exists() and (not images_root.is_dir() or any(images_root.iterdir())):
            raise ValueError("evaluation images output must be a new or empty directory; choose another --images-out")
        filenames = [f"{index:06d}_{Path(entry['name']).stem}.png"
                     for index, entry in enumerate(dataset.entries)]
        image_artifacts = {images_root / name for name in filenames} | {images_root / "comparison.json"}
        if destination is not None and (destination in image_artifacts or destination == images_root
                                        or destination in images_root.parents):
            raise ValueError("evaluation report would overwrite comparison output")
    net, metadata = load_export(source, checkpoint_path)
    if (dataset.width, dataset.height) != (metadata["width"], metadata["height"]):
        raise ValueError("evaluation cache dimensions differ from the distilled model")
    if images_root is not None:
        images_root.mkdir(parents=True, exist_ok=True)
        comparisons = {"status": "running", "directory": str(images_root),
                       "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
                       "side_by_side_order": ["input", "teacher", "student"], "images": []}
        _write_json(images_root / "comparison.json", comparisons)
        log(f"Saving evaluation comparisons to {images_root}: input | teacher | student")

    def record_image(index, name, input_frame, target_frame, output_frame):
        if images_root is not None:
            path = images_root / filenames[index]
            pixels = np.concatenate((input_frame, target_frame, output_frame), axis=1)
            _atomic_write(path, lambda temporary: Image.fromarray(pixels).save(temporary, format="PNG"))
            comparisons["images"].append({"name": name, "file": path.name})
            _write_json(images_root / "comparison.json", comparisons)
            log(f"Comparison {index + 1}/{len(dataset)} saved: {path}")
        if on_image is not None:
            on_image(index, name, input_frame, target_frame, output_frame)

    try:
        metrics = evaluate_network(net, dataset, device, log,
                                   on_image=record_image if images_root is not None else on_image)
    except (Exception, KeyboardInterrupt) as error:
        if comparisons is not None:
            comparisons["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
            comparisons["error"] = f"{type(error).__name__}: {error}"
            _write_json(images_root / "comparison.json", comparisons)
        raise
    result = {"metadata": metadata, "metrics": metrics}
    if comparisons is not None:
        comparisons["status"] = "completed"
        _write_json(images_root / "comparison.json", comparisons)
        result["comparisons"] = comparisons
    if destination is not None:
        _write_json(destination, result)
    return result


def _write_training_report(destination: Path, report: dict) -> None:
    _write_json(destination / "report.json", report)
    lines = ["# Recovery distillation", "", f"Status: {report['status']}",
             f"Remaining MACs: {report['metadata']['pruning']['macs'] / 1e9:.6f} GMAC "
             f"({report['metadata']['pruning']['keep_ratio'] * 100:.4f}%)",
             f"Dropped blocks: `{report['metadata']['dropped_blocks']}`", "",
             "All validation scores use the ordinary inference graph with materialized weights.",
             "Float PSNR uses [0, 1]; uint8 PSNR and SSIM use the original output rounding.", ""]
    if report.get("error"):
        lines += [f"Error: {report['error']}", ""]
    baseline = report.get("baseline")
    if baseline:
        lines += [f"Before distillation: PSNR={baseline['psnr_db']:.4f} dB, "
                  f"uint8 PSNR={baseline['psnr_uint8_db']:.4f} dB", ""]
    best = report.get("export_validation") or report.get("best")
    if best:
        improved = baseline is not None and best["mse"] < baseline["mse"]
        lines += [f"Best exported result: PSNR={best['psnr_db']:.4f} dB, "
                  f"uint8 PSNR={best['psnr_uint8_db']:.4f} dB; "
                  f"improved over pruning-only baseline: {improved}", ""]
    lines += ["| Epoch | Training loss | Validation loss | Validation MSE | Float PSNR | uint8 PSNR | SSIM | Learning rate |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for epoch in report["history"]:
        metrics = epoch["validation"]
        structure = "N/A" if metrics["ssim"] is None else f"{metrics['ssim']:.6f}"
        validation_loss = "N/A" if metrics.get("loss") is None else f"{metrics['loss']:.8g}"
        lines.append(f"| {epoch['epoch']} | {epoch['loss']:.8g} | {validation_loss} | {metrics['mse']:.8g} | "
                     f"{metrics['psnr_db']:.4f} | {metrics['psnr_uint8_db']:.4f} | "
                     f"{structure} | {epoch['learning_rate']:.8g} |")
    lines += ["", f"Peak GPU allocation: {report.get('peak_memory_bytes', 0) / 2**30:.3f} GiB",
              f"Elapsed: {report.get('elapsed_seconds', 0):.2f} s", "",
              "Resume checkpoints are saved at completed epoch boundaries. "
              "See report.json for per-image scores, metadata and preflight memory.", ""]
    _atomic_write(destination / "report.md", lambda temporary: temporary.write_text("\n".join(lines), encoding="utf-8"))


def train_distillation(weights_path, pruning_report, train_cache, val_cache, out_dir,
                       device="auto", config: TrainConfig | None = None, resume=None, log=print,
                       *, tensorboard: bool = False) -> dict:
    config = config or TrainConfig()
    config.validate()
    require_reference()
    writer_type = load_summary_writer() if tensorboard else None
    source, destination = Path(weights_path).expanduser().resolve(), Path(out_dir).expanduser().resolve()
    weights_hash = file_sha256(source)
    log("Validating training cache, source hashes and floating-point targets")
    train = CachedFrames(train_cache, weights_hash)
    log("Validating held-out cache, source hashes and floating-point targets")
    validation = CachedFrames(val_cache, weights_hash)
    _check_data(train, validation)
    pruning = read_pruning_report(pruning_report, build_plan(train.width, train.height))
    if destination == train.root or destination == validation.root or destination in (source, Path(pruning_report).resolve()):
        raise ValueError("training output must be separate from source weights, pruning report and caches")
    if destination.exists() and not destination.is_dir():
        raise ValueError("training output must be a directory")
    if resume is None and destination.exists() and any(destination.iterdir()):
        raise ValueError("training output is not empty; use --resume or a new output directory")
    destination.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(device))
    metadata = {"schema_version": 1, "weights_sha256": weights_hash, "weights_path": str(source),
                "width": train.width, "height": train.height, "controls": asdict(Controls()),
                "dropped_blocks": pruning["blocks"], "pruning": pruning,
                "train_manifest_sha256": file_sha256(train.root / "manifest.json"),
                "val_manifest_sha256": file_sha256(validation.root / "manifest.json"),
                "train_cache": str(train.root), "val_cache": str(validation.root),
                "config": asdict(config), "device": str(device), "torch_version": str(torch.__version__)}
    restored = torch.load(str(resume), map_location="cpu", weights_only=True) if resume is not None else None
    if restored is not None:
        if restored.get("schema_version") != 1:
            raise ValueError("unsupported training checkpoint")
        _validate_metadata(restored.get("metadata"), weights_hash)
        for field in ("width", "height", "controls", "dropped_blocks", "train_manifest_sha256",
                      "val_manifest_sha256", "config"):
            if restored["metadata"].get(field) != metadata[field]:
                raise ValueError(f"resume checkpoint has different {field}")
        # last.pt refers to the best artifact in the same run directory.
        if Path(resume).expanduser().resolve().parent != destination:
            raise ValueError("resume in the checkpoint's original output directory to preserve best artifacts")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    log(f"Building student: {train.width}x{train.height}, "
        f"{pruning['keep_ratio'] * 100:.4f}% MACs; dropped blocks={pruning['blocks']}")
    student = DistilledStudent(_new_network(source, train.width, train.height), pruning["blocks"],
                               checkpoint_blocks=config.checkpoint_blocks).to(device)
    inference = _new_network(source, train.width, train.height)
    inference.skip = set(pruning["layer_indices"])
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.min_learning_rate)
    report = {"status": "running", "metadata": metadata, "baseline": None, "best": None,
              "best_epoch": 0, "history": [], "global_step": 0, "completed_epochs": 0,
              "preflight_peak_memory_bytes": 0, "peak_memory_bytes": 0}
    completed, bad_epochs = 0, 0
    if restored is not None:
        student.load_state_dict(restored["student_state"], strict=True)
        optimizer.load_state_dict(restored["optimizer_state"])
        scheduler.load_state_dict(restored["scheduler_state"])
        report.update(restored["report"])
        report.update(status="running", metadata=metadata)
        report.pop("error", None)
        completed, bad_epochs = restored["completed_epochs"], restored["bad_epochs"]
        torch.set_rng_state(restored["rng_state"])
        if device.type == "cuda" and restored.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(restored["cuda_rng_state"])
        log(f"Resuming at epoch {completed + 1}, optimizer step {report['global_step']}")
    started = time.perf_counter()
    previous_elapsed = report.get("elapsed_seconds", 0.0)

    def persist():
        report["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
        if device.type == "cuda":
            report["peak_memory_bytes"] = max(report["peak_memory_bytes"], torch.cuda.max_memory_allocated(device))
        _write_training_report(destination, report)

    def save_checkpoint(path):
        payload = {"schema_version": 1, "metadata": metadata, "student_state": student.state_dict(),
                   "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                   "completed_epochs": completed, "bad_epochs": bad_epochs, "report": report,
                   "rng_state": torch.get_rng_state(),
                   "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        _atomic_write(path, lambda temporary: torch.save(payload, temporary))

    writer = None

    def validate_student():
        inference.load_state_dict(student.export_state(), strict=True)
        return evaluate_network(inference, validation, device, log,
                                on_image=image_callback(writer, "val", report["global_step"]))

    try:
        if writer_type is not None:
            writer = writer_type(log_dir=str(destination / "tensorboard"), flush_secs=30,
                                 purge_step=report["global_step"] + 1 if restored is not None else None)
            log(f"TensorBoard: {destination / 'tensorboard'} (optimizer steps; images: input | teacher | student)")
        persist()
        if restored is None:
            log("Evaluating pruning-only baseline through the ordinary inference graph")
            report["baseline"] = report["best"] = validate_student()
            persist()
            save_checkpoint(destination / "best.pt")
            save_checkpoint(destination / "last.pt")
            _export(student, metadata, destination / "best.safetensors")
            write_metrics(writer, "val", report["baseline"], 0)
            if writer is not None:
                writer.flush()
        # No optimizer update: verify full-frame backward and memory before the real run.
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        log("Preflight: full-frame forward and backward, without updating weights")
        student.train()
        first = train[0]
        prediction = student(frame_to_tensor(first["input"], device))
        loss = distillation_loss(prediction, first["target"].to(device))
        if not torch.isfinite(loss).item():
            raise ValueError("preflight loss is nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip, error_if_nonfinite=True)
        optimizer.zero_grad(set_to_none=True)
        del prediction, loss, first
        if device.type == "cuda":
            report["preflight_peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        log(f"Preflight complete; peak allocation={report['preflight_peak_memory_bytes'] / 2**30:.3f} GiB")
        persist()

        for epoch in range(completed, config.epochs):
            if bad_epochs >= config.patience:
                break
            student.train()
            optimizer.zero_grad(set_to_none=True)
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(config.seed + epoch)).tolist()
            loss_sum, learning_rate = 0.0, optimizer.param_groups[0]["lr"]
            group_loss = 0.0
            epoch_started = time.perf_counter()
            for position, index in enumerate(order):
                item = train[index]
                prediction = student(frame_to_tensor(item["input"], device))
                loss = distillation_loss(prediction, item["target"].to(device))
                loss_value = float(loss.detach())
                if not math.isfinite(loss_value):
                    raise ValueError("training loss is nonfinite")
                loss_sum += loss_value
                group_loss += loss_value
                group_start = position - position % config.accumulation
                group_size = min(config.accumulation, len(train) - group_start)
                (loss / group_size).backward()
                if (position + 1) % config.accumulation == 0 or position + 1 == len(order):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        student.parameters(), config.grad_clip, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    report["global_step"] += 1
                    write_scalar(writer, "train/loss_step", group_loss / group_size, report["global_step"])
                    write_scalar(writer, "train/learning_rate", learning_rate, report["global_step"])
                    write_scalar(writer, "train/grad_norm", grad_norm, report["global_step"])
                    group_loss = 0.0
                log(f"Epoch {epoch + 1}/{config.epochs} image {position + 1}/{len(order)} "
                    f"{item['name']}: loss={loss_value:.8g}, step={report['global_step']}, lr={learning_rate:.4g}")
                del prediction, loss, item
            metrics = validate_student()
            improved = metrics["mse"] < report["best"]["mse"]
            if improved:
                report["best"], report["best_epoch"], bad_epochs = metrics, epoch + 1, 0
            else:
                bad_epochs += 1
            completed = epoch + 1
            report["completed_epochs"] = completed
            report["history"].append({"epoch": completed, "loss": loss_sum / len(train),
                                      "learning_rate": learning_rate, "validation": metrics,
                                      "seconds": time.perf_counter() - epoch_started})
            scheduler.step()
            persist()
            if improved:
                save_checkpoint(destination / "best.pt")
                _export(student, metadata, destination / "best.safetensors")
            save_checkpoint(destination / "last.pt")
            step = report["global_step"]
            write_scalar(writer, "train/loss_epoch", loss_sum / len(train), step)
            write_scalar(writer, "train/epoch", completed, step)
            write_metrics(writer, "val", metrics, step)
            write_scalar(writer, "best/mse", report["best"]["mse"], step)
            write_scalar(writer, "best/epoch", report["best_epoch"], step)
            write_scalar(writer, "val/bad_epochs", bad_epochs, step)
            if writer is not None:
                writer.flush()
            log(f"Epoch {completed}: validation PSNR={metrics['psnr_db']:.4f}dB, "
                f"best={report['best']['psnr_db']:.4f}dB, no-improvement epochs={bad_epochs}")
        # Validate the actual deployable artifact, not the training surrogate path.
        log("Reloading best.safetensors into the ordinary inference model")
        report["export_validation"] = evaluate_export(
            source, destination / "best.safetensors", val_cache, device=device, log=log,
            on_image=image_callback(writer, "final", report["global_step"]))["metrics"]
        write_metrics(writer, "final", report["export_validation"], report["global_step"])
        write_scalar(writer, "final/best_epoch", report["best_epoch"], report["global_step"])
        report["improved"] = report["export_validation"]["mse"] < report["baseline"]["mse"]
        report["status"] = "completed"
        report["stop_reason"] = "early_stopping" if bad_epochs >= config.patience else "epochs_completed"
        persist()
        return report
    except (Exception, KeyboardInterrupt) as error:
        optimizer.zero_grad(set_to_none=True)
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        report["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, torch.cuda.OutOfMemoryError):
            report["error"] += "; no automatic resolution change was made"
        persist()
        raise
    finally:
        if writer is not None:
            active_error = sys.exception()
            try:
                writer.close()
            except Exception as close_error:
                if active_error is None:
                    raise
                active_error.add_note(f"TensorBoard close also failed: {close_error}")
