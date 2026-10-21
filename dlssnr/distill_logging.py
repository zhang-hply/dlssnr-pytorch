"""Optional TensorBoard output for recovery distillation."""
from __future__ import annotations

import math

import numpy as np


def load_summary_writer():
    # Keep inference and training without --tensorboard independent of this extra.
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard logging requires the optional tensorboard dependency; "
            "install it with `uv pip install tensorboard` or add --extra tensorboard to uv sync"
        ) from error
    return SummaryWriter


def write_scalar(writer, tag: str, value, step: int) -> None:
    if writer is None or value is None:
        return
    value = float(value)
    if math.isfinite(value):
        writer.add_scalar(tag, value, step)
    else:
        writer.add_text(f"{tag}/nonfinite", str(value), step)


def write_metrics(writer, prefix: str, metrics: dict, step: int) -> None:
    for key in ("loss", "mse", "psnr_db", "mse_uint8", "psnr_uint8_db", "ssim"):
        write_scalar(writer, f"{prefix}/{key}", metrics.get(key), step)


def image_callback(writer, prefix: str, step: int):
    """Stream the first three full-resolution comparisons without retaining frames."""
    if writer is None:
        return None

    def record(index, name, input_frame, target_frame, output_frame):
        if index >= 3:
            return
        tag = f"{prefix}/images/{index:03d}"
        writer.add_text(f"{tag}/description", f"{name}\n\nLeft to right: input | teacher | student", step)
        comparison = np.concatenate((input_frame, target_frame, output_frame), axis=1)
        writer.add_image(tag, comparison, step, dataformats="HWC")

    return record
