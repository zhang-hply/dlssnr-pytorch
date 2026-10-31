"""Differentiable image and teacher-feature objectives, evaluated in FP32."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F


FEATURE_BLOCKS = (14, 22, 39, 56)


@dataclass(frozen=True)
class LossWeights:
    rgb_weight: float = 1.0
    edge_weight: float = 0.1
    ssim_weight: float = 0.1
    multiscale_weight: float = 0.1
    feature_weight: float = 0.05

    def validate(self):
        values = asdict(self).values()
        if any(not math.isfinite(v) or v < 0 for v in values) or not any(values):
            raise ValueError("loss weights must be finite, nonnegative, and not all zero")

    @classmethod
    def from_metadata(cls, config):
        # Missing keys identify the original RGB-plus-edge objective, not new defaults.
        legacy = asdict(LEGACY_WEIGHTS)
        return cls(**{key: config.get(key, value) for key, value in legacy.items()})


LEGACY_WEIGHTS = LossWeights(ssim_weight=0.0, multiscale_weight=0.0, feature_weight=0.0)


def structural_loss(x, y):
    """1 - RGB SSIM; replicate padding also supports one-pixel dimensions."""
    axis = torch.arange(11, device=x.device, dtype=torch.float32) - 5
    kernel = torch.exp(-0.5 * (axis / 1.5).square())
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    kh = kernel.view(1, 1, 11, 1).expand(channels, 1, 11, 1)
    kw = kernel.view(1, 1, 1, 11).expand(channels, 1, 1, 11)

    def blur(z):
        z = F.conv2d(F.pad(z, (0, 0, 5, 5), mode="replicate"), kh, groups=channels)
        return F.conv2d(F.pad(z, (5, 5, 0, 0), mode="replicate"), kw, groups=channels)

    mx, my = blur(x), blur(y)
    vx = (blur(x.square()) - mx.square()).clamp_min(0)
    vy = (blur(y.square()) - my.square()).clamp_min(0)
    cov = blur(x * y) - mx * my
    # Roundoff can make constant-image covariance negative while variances were
    # clamped to zero. Enforce the arithmetic-mean variance bound, avoiding a
    # sqrt derivative at zero and keeping identical constant images at SSIM=1.
    bound = (vx + vy) / 2
    cov = torch.maximum(torch.minimum(cov, bound), -bound)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mx * my + c1) * (2 * cov + c2)) / (
        (mx.square() + my.square() + c1) * (vx + vy + c2))
    return 1 - score.clamp(-1, 1).mean()


def loss_components(prediction, target, weights=LEGACY_WEIGHTS, *, features=None,
                    teacher_features=None, allow_missing_features=False):
    """Return raw enabled terms, weighted image subtotal, and weighted total.

    Disabled terms are zero and incur no computation. An unavailable, enabled
    feature term and total are None, only for standalone RGB-only evaluation.
    """
    weights.validate()
    if prediction.shape != target.shape or prediction.ndim != 4 or any(n < 1 for n in prediction.shape):
        raise ValueError("prediction and target must have matching NCHW shapes")
    x, y = prediction.float(), target.detach().to(prediction.device, dtype=torch.float32)
    zero = x.sum() * 0
    terms = {name: zero for name in ("rgb", "edge", "ssim", "multiscale", "feature")}
    if weights.rgb_weight:
        terms["rgb"] = (x - y).abs().mean()
    if weights.edge_weight:
        horizontal = ((x[..., 1:] - x[..., :-1]) - (y[..., 1:] - y[..., :-1])).abs().mean() \
            if x.shape[-1] > 1 else zero
        vertical = ((x[..., 1:, :] - x[..., :-1, :]) - (y[..., 1:, :] - y[..., :-1, :])).abs().mean() \
            if x.shape[-2] > 1 else zero
        terms["edge"] = (horizontal + vertical) / 2
    if weights.ssim_weight:
        terms["ssim"] = structural_loss(x, y)
    if weights.multiscale_weight:
        levels = []
        for scale in (2, 4, 8):
            size = tuple(max(1, n // scale) for n in x.shape[-2:])
            levels.append((F.interpolate(x, size=size, mode="area") -
                           F.interpolate(y, size=size, mode="area")).abs().mean())
        terms["multiscale"] = torch.stack(levels).mean()
    image_loss = sum(getattr(weights, name + "_weight") * terms[name]
                     for name in ("rgb", "edge", "ssim", "multiscale"))
    if weights.feature_weight:
        if features is None or teacher_features is None:
            if not allow_missing_features:
                raise ValueError("feature supervision requires teacher features; rerun prepare for both caches")
            terms["feature"] = None
        else:
            if set(features) != set(FEATURE_BLOCKS) or set(teacher_features) != set(FEATURE_BLOCKS):
                raise ValueError(f"feature supervision requires blocks {FEATURE_BLOCKS}")
            losses = []
            for block in FEATURE_BLOCKS:
                student = features[block].float()
                teacher = teacher_features[block].detach().to(student.device, dtype=torch.float32)
                if student.shape != teacher.shape or student.ndim != 4 or student.numel() == 0:
                    raise ValueError(f"feature shape mismatch at block {block}")
                scale = teacher.square().mean().sqrt().clamp_min(1e-3)
                losses.append((student - teacher).abs().mean() / scale)
            terms["feature"] = torch.stack(losses).mean()
    terms["image_loss"] = image_loss
    terms["loss"] = None if terms["feature"] is None else image_loss + weights.feature_weight * terms["feature"]
    return terms
