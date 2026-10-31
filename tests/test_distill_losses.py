"""Numerical behavior and gradient coverage for the combined objective."""
from dataclasses import asdict, replace

import pytest
import torch

from dlssnr.distill_losses import FEATURE_BLOCKS, LEGACY_WEIGHTS, LossWeights, loss_components


@pytest.mark.parametrize("shape", [(1, 3, 1, 1), (1, 3, 1, 9), (1, 3, 7, 1), (1, 3, 13, 17)])
def test_identical_images_have_zero_loss_and_finite_gradients(shape):
    x = torch.rand(shape, requires_grad=True)
    features = {block: torch.rand(1, 2, 3, 5, requires_grad=True) for block in FEATURE_BLOCKS}
    target = x.detach().clone().requires_grad_()
    teacher = {block: value.detach().clone().requires_grad_() for block, value in features.items()}
    terms = loss_components(x, target, LossWeights(), features=features, teacher_features=teacher)
    assert all(float(v.detach()) == pytest.approx(0, abs=2e-6) for v in terms.values())
    terms["loss"].backward()
    assert torch.isfinite(x.grad).all() and target.grad is None
    assert all(torch.isfinite(v.grad).all() for v in features.values())
    assert all(v.grad is None for v in teacher.values())


@pytest.mark.parametrize("term", ["rgb", "edge", "ssim", "multiscale", "feature"])
def test_each_enabled_term_supplies_nonzero_finite_gradients(term):
    weights = LossWeights(**{key: float(key == term + "_weight") for key in asdict(LossWeights())})
    x = torch.rand(1, 3, 13, 17, requires_grad=True)
    features = {block: torch.full((1, 2, 3, 5), 0.5, requires_grad=True) for block in FEATURE_BLOCKS}
    teacher = {block: torch.zeros_like(value) for block, value in features.items()}
    terms = loss_components(x, torch.zeros_like(x), weights, features=features, teacher_features=teacher)
    assert terms["loss"] > 0
    terms["loss"].backward()
    grads = [v.grad for v in features.values()] if term == "feature" else [x.grad]
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in grads)


def test_feature_normalization_and_weighted_total():
    x = torch.full((1, 3, 5, 7), 0.4, requires_grad=True)
    features = {block: torch.full((1, 2, 2, 2), (i + 1) * 3.0) for i, block in enumerate(FEATURE_BLOCKS)}
    teacher = {block: value / 3 for block, value in features.items()}
    weights = LossWeights()
    terms = loss_components(x, x * 0.5, weights, features=features, teacher_features=teacher)
    assert terms["feature"] == 2
    expected = sum(getattr(weights, key + "_weight") * terms[key]
                   for key in ("rgb", "edge", "ssim", "multiscale", "feature"))
    torch.testing.assert_close(terms["loss"], expected)


def test_disabled_losses_are_not_computed_and_missing_features_are_explicit(monkeypatch):
    import dlssnr.distill_losses as losses

    def fail(*args):
        raise AssertionError("disabled SSIM should not be computed")

    monkeypatch.setattr(losses, "structural_loss", fail)
    x = torch.rand(1, 3, 3, 4)
    terms = loss_components(x, x, LEGACY_WEIGHTS)
    assert terms["ssim"] == terms["multiscale"] == terms["feature"] == 0
    weights = replace(LEGACY_WEIGHTS, feature_weight=0.05)
    with pytest.raises(ValueError, match="rerun prepare"):
        loss_components(x, x, weights)
    terms = loss_components(x, x, weights, allow_missing_features=True)
    assert terms["feature"] is None and terms["loss"] is None and terms["image_loss"] == 0


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_weights(value):
    with pytest.raises(ValueError, match="loss weights"):
        replace(LossWeights(), feature_weight=value).validate()


def test_zero_objective_and_bad_feature_shapes_rejected():
    with pytest.raises(ValueError, match="not all zero"):
        LossWeights(**dict.fromkeys(asdict(LossWeights()), 0)).validate()
    x = torch.rand(1, 3, 3, 4)
    features = dict.fromkeys(FEATURE_BLOCKS, x)
    with pytest.raises(ValueError, match="requires blocks"):
        loss_components(x, x, LossWeights(), features={}, teacher_features=features)
    with pytest.raises(ValueError, match="feature shape mismatch"):
        loss_components(x, x, LossWeights(), features=features,
                        teacher_features=dict.fromkeys(FEATURE_BLOCKS, x[..., :1]))
