"""Opt-in exact-forward numerical surrogates for recovery distillation.

The inference path keeps the original arithmetic. Inside ``distillation_mode``
rounding has a straight-through gradient, while integer exponent construction
and half-precision normalisation use explicitly defined float32 gradients.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import functools
import math

import torch

_DISTILLING: ContextVar[bool] = ContextVar("dlssnr_distilling", default=False)


@contextmanager
def distillation_mode():
    """Enable differentiable reference arithmetic, including recomputation."""
    token = _DISTILLING.set(True)
    try:
        yield
    finally:
        _DISTILLING.reset(token)


def is_distilling() -> bool:
    return _DISTILLING.get()


def inference_forward(function):
    """Keep inference detached; in distillation respect the caller's grad mode."""
    @functools.wraps(function)
    def forward(*args, **kwargs):
        if is_distilling():
            return function(*args, **kwargs)
        with torch.no_grad():
            return function(*args, **kwargs)
    return forward


class _RoundFP8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.float()
        ctx.save_for_backward(x)
        return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        return torch.where((x >= -448.0) & (x <= 448.0), gradient.float(), 0.0)


class _RoundFP16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.to(torch.float16).float()

    @staticmethod
    def backward(ctx, gradient):
        return gradient.float()


def fp8_ste(x: torch.Tensor) -> torch.Tensor:
    """Saturating e4m3 rounding, identity gradient inside the representable range."""
    return _RoundFP8.apply(x)


def fp16_ste(x: torch.Tensor) -> torch.Tensor:
    """Exact half rounding with float32 storage and a straight-through gradient."""
    return _RoundFP16.apply(x)


class _Surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, exact, surrogate):
        return exact.float()

    @staticmethod
    def backward(ctx, gradient):
        return None, gradient.float()


def with_surrogate(exact: torch.Tensor, surrogate: torch.Tensor) -> torch.Tensor:
    """Return exact bits and differentiate only the float32 surrogate expression.

    Unlike ``proxy + (exact - proxy).detach()``, this remains bit exact and does
    not suffer cancellation or inf-minus-inf in the forward pass.
    """
    return _Surrogate.apply(exact.detach(), surrogate)


class _RampExponent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, exact, ramp, low, high, rate):
        ctx.save_for_backward(exact, ramp)
        ctx.low, ctx.high, ctx.rate = low, high, rate
        return exact.float()

    @staticmethod
    def backward(ctx, gradient):
        exact, ramp = ctx.saved_tensors
        inside = (ramp >= ctx.low) & (ramp <= ctx.high)
        derivative = exact.float() * (math.log(2.0) * ctx.rate)
        return None, torch.where(inside, gradient.float() * derivative, 0.0), None, None, None


def ramp_exponential(exact: torch.Tensor, ramp: torch.Tensor,
                     low: float, high: float, rate: float) -> torch.Tensor:
    """Exact bit exponent with a continuous exponential proxy in ramp space.

    The original bit shift advances 32 exponents per ramp unit for window
    attention and 16 for ViT. The upstream ramp supplies its family-specific
    logit slope and bias gradient; values outside the original clamp get zero.
    """
    return _RampExponent.apply(exact.detach(), ramp.float(), low, high, rate)
