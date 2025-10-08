from __future__ import annotations

from typing import Optional

import torch

__all__ = ["linear_w4a8", "__version__"]

__version__ = "0.0.0-fallback"


def _unpack_int4(packed_w: torch.Tensor, out_aligned: int, in_aligned: int) -> torch.Tensor:
    """Unpack uint8 nibble-packed INT4 tensor into int8 values in [-8, 7]."""
    if packed_w.dtype != torch.uint8:
        raise ValueError("packed_w must be a uint8 tensor.")
    low = (packed_w & 0xF).to(torch.int8)
    high = ((packed_w >> 4) & 0xF).to(torch.int8)
    stacked = torch.stack((low, high), dim=-1).reshape(packed_w.shape[0], -1)
    stacked = torch.where(stacked >= 8, stacked - 16, stacked)
    return stacked[:, :in_aligned]


def linear_w4a8(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    s_w: torch.Tensor,
    s_a: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fallback BitBLAS linear kernel implemented with PyTorch ops.

    This simulates the accumulation logic of a W4A8 GEMM to provide numerical
    parity checks when the native BitBLAS library is unavailable.
    """
    if x.ndim < 2:
        raise ValueError("Input x must have at least 2 dimensions.")

    orig_shape = x.shape
    flat_x = x.reshape(-1, orig_shape[-1]).to(torch.float32)
    out_aligned, half_cols = packed_w.shape
    in_aligned = half_cols * 2

    if flat_x.shape[1] < in_aligned:
        pad = in_aligned - flat_x.shape[1]
        flat_x = torch.nn.functional.pad(flat_x, (0, pad), value=0.0)

    s_w = s_w.to(device=x.device, dtype=torch.float32)
    s_a = s_a.to(device=x.device, dtype=torch.float32)

    q_w = _unpack_int4(packed_w.to(device=x.device), out_aligned, in_aligned).to(torch.float32)

    inv_scale = torch.clamp(s_a, min=1e-6).reciprocal()
    x_q = torch.clamp(torch.round(flat_x * inv_scale), -128, 127)
    dequant_x = x_q * s_a

    acc = torch.matmul(dequant_x, q_w.t())
    acc = acc * s_w

    if bias is not None:
        acc = acc + bias.to(device=acc.device, dtype=acc.dtype)

    acc = acc.reshape(*orig_shape[:-1], out_aligned)
    return acc.to(dtype=x.dtype)
