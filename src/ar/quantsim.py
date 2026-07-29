from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor

SUPPORTED_BITS: tuple[int, ...] = (4, 8)


class QuantConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    bits: int = Field(description="Integer width. 4 or 8.")
    group_size: int = Field(
        description="Weights per quantization group along the input dimension. "
        "-1 means one group per output row (per-channel)."
    )
    scheme: Literal["asymmetric", "symmetric", "symmetric_gptq"] = "asymmetric"
    """
    asymmetric      GPTQ convention. Bit-exact against gptqmodel (EXP-003).
    symmetric       Signed codes in [-2^(b-1), 2^(b-1)-1], scale = absmax/(2^(b-1)-1).
                    The AWQ/torch-style convention. Does NOT match gptqmodel.
    symmetric_gptq  gptqmodel's own sym convention: unsigned codes with a fixed
                    zero point at (2^b)/2 and scale = (xmax-xmin)/(2^b-1) after
                    mirroring the range. Bit-exact against gptqmodel (EXP-003).
                    Use this when reporting numbers meant to describe what a
                    gptqmodel-quantized checkpoint actually contains.
    """

    @field_validator("bits")
    @classmethod
    def _check_bits(cls, v: int) -> int:
        if v not in SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {v}")
        return v

    @field_validator("group_size")
    @classmethod
    def _check_group_size(cls, v: int) -> int:
        if v == 0 or v < -1:
            raise ValueError(f"group_size must be a positive int or -1, got {v}")
        return v

    @property
    def name(self) -> str:
        g = "per_channel" if self.group_size == -1 else f"g{self.group_size}"
        s = {"asymmetric": "asym", "symmetric": "sym", "symmetric_gptq": "symgptq"}[
            self.scheme
        ]
        return f"int{self.bits}_{g}_{s}"


class QuantResult(NamedTuple):
    codes: Tensor
    """Integer codes, same shape as the input weight."""
    dequant: Tensor
    """Quantized-then-dequantized weight, same shape and dtype as compute dtype."""
    scale: Tensor
    """Per-group step size, shape (out_features, n_groups)."""
    zero: Tensor | None
    """Per-group integer zero point for asymmetric, None for symmetric."""

    def step_per_weight(self, group_size: int) -> Tensor:
        """Broadcast the per-group step size back to one value per weight.

        Needed for the step-ratio distribution |delta| / (s/2), which is defined
        per weight against the step size of the group that weight lives in.
        """
        n_in = self.dequant.shape[1]
        g = n_in if group_size == -1 else group_size
        return self.scale.repeat_interleave(g, dim=1)[:, :n_in]


def _q_range(bits: int, scheme: str) -> tuple[int, int]:
    if scheme == "symmetric":
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def _pad_to_groups(w: Tensor, group_size: int) -> tuple[Tensor, int, int]:
    """Pad the input dimension with NaN so it divides evenly into groups.

    NaN is the padding sentinel because it is excluded from both min and max via
    an explicit mask below; padding with a real value would corrupt the group's
    range and silently change its step size. Real NaNs in the input are rejected
    up front so the sentinel is never ambiguous.
    """
    n_out, n_in = w.shape
    g = n_in if group_size == -1 else group_size
    pad = (-n_in) % g
    if pad:
        filler = torch.full((n_out, pad), float("nan"), dtype=w.dtype, device=w.device)
        w = torch.cat([w, filler], dim=1)
    return w, g, pad


def quantize_dequantize(
    weight: Tensor,
    config: QuantConfig,
    compute_dtype: torch.dtype = torch.float32,
) -> QuantResult:
    """Group-wise affine quantize-then-dequantize with explicit step sizes.

    Asymmetric mode follows the GPTQ convention: the group range is clamped to
    include zero (xmax >= 0 >= xmin) so that zero is always exactly
    representable, and an all-zero group is assigned the range [-1, 1] rather
    than a zero step size. Matching that convention is what makes the
    cross-check against gptqmodel meaningful.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight matrix, got shape {tuple(weight.shape)}")
    if torch.isnan(weight).any():
        raise ValueError(
            "Input weight contains NaN, which collides with the group-padding "
            "sentinel. Refusing to quantize."
        )
    if torch.isinf(weight).any():
        raise ValueError("Input weight contains inf. Refusing to quantize.")

    w = weight.to(compute_dtype)
    n_out, n_in = w.shape
    w_pad, g, pad = _pad_to_groups(w, config.group_size)
    n_groups = w_pad.shape[1] // g

    view = w_pad.reshape(n_out * n_groups, g)
    valid = ~torch.isnan(view)

    neg_inf = torch.tensor(float("-inf"), dtype=compute_dtype, device=w.device)
    pos_inf = torch.tensor(float("inf"), dtype=compute_dtype, device=w.device)
    qmin, qmax = _q_range(config.bits, config.scheme)

    if config.scheme in ("asymmetric", "symmetric_gptq"):
        xmax = torch.where(valid, view, neg_inf).amax(dim=-1)
        xmin = torch.where(valid, view, pos_inf).amin(dim=-1)
        xmax = xmax.clamp(min=0.0)
        xmin = xmin.clamp(max=0.0)
        if config.scheme == "symmetric_gptq":
            # Mirror the range, but only where xmin is strictly negative. gptqmodel
            # leaves xmin at 0 for an all-non-negative group, which combined with a
            # fixed zero point at (qmax+1)/2 clips the group's upper half. Faithful
            # to the reference: the point of this scheme is to match what the
            # toolchain actually produces, quirks included.
            xmax = torch.maximum(xmin.abs(), xmax)
            xmin = torch.where(xmin < 0.0, -xmax, xmin)
        degenerate = (xmax == 0.0) & (xmin == 0.0)
        xmax = torch.where(degenerate, torch.ones_like(xmax), xmax)
        xmin = torch.where(degenerate, -torch.ones_like(xmin), xmin)
        scale = (xmax - xmin) / qmax
        if config.scheme == "symmetric_gptq":
            zero = torch.full_like(scale, (qmax + 1) // 2)
        else:
            zero = torch.round(-xmin / scale)
        codes = torch.clamp(
            torch.round(view / scale.unsqueeze(-1)) + zero.unsqueeze(-1), qmin, qmax
        )
        deq = (codes - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
    else:
        absmax = torch.where(valid, view.abs(), neg_inf).amax(dim=-1)
        absmax = torch.where(absmax == 0.0, torch.ones_like(absmax), absmax)
        scale = absmax / qmax
        zero = None
        codes = torch.clamp(torch.round(view / scale.unsqueeze(-1)), qmin, qmax)
        deq = codes * scale.unsqueeze(-1)

    codes = codes.reshape(n_out, n_groups * g)
    deq = deq.reshape(n_out, n_groups * g)
    if pad:
        codes = codes[:, :n_in]
        deq = deq[:, :n_in]

    return QuantResult(
        codes=codes,
        dequant=deq,
        scale=scale.reshape(n_out, n_groups),
        zero=None if zero is None else zero.reshape(n_out, n_groups),
    )
