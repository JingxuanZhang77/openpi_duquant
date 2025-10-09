"""BitBLAS W4A8 kernel utilities for DuQuant quantization."""
from __future__ import annotations

import functools
import logging
import os
from typing import Callable, Optional

import torch

__all__ = ["ensure_bitblas_linear_kernel"]

_KERNEL_CACHE: Optional[Callable] = None


def ensure_bitblas_linear_kernel() -> Optional[Callable]:
    """
    Returns the BitBLAS W4A8 linear kernel function if available, otherwise None.

    The kernel signature is:
        linear_w4a8(
            x: torch.Tensor,          # [batch, in_features_aligned] float16/bfloat16
            packed_w: torch.Tensor,   # [out_features_aligned, in_features_aligned // 2] uint8
            s_w: torch.Tensor,        # [out_features_aligned] float16
            s_a: torch.Tensor,        # [in_features_aligned] float16
            bias: Optional[torch.Tensor] = None,  # [out_features_aligned] float16
        ) -> torch.Tensor:            # [batch, out_features_aligned] same dtype as x
    """
    global _KERNEL_CACHE

    if _KERNEL_CACHE is not None:
        return _KERNEL_CACHE

    try:
        # For now, use the optimized PyTorch fallback implementation
        # True BitBLAS INT4 kernels require weight format conversion which
        # needs additional work to match DuQuant's packing format
        from . import bitblas_fallback

        logging.info("[BITBLAS] Using optimized PyTorch W4A8 fallback implementation")

        _KERNEL_CACHE = bitblas_fallback.linear_w4a8
        return bitblas_fallback.linear_w4a8

    except ImportError:
        logging.warning("[BITBLAS] BitBLAS not available, using fallback implementation")
        _KERNEL_CACHE = None
        return None
    except Exception as exc:
        logging.warning(f"[BITBLAS] Failed to initialize kernel: {exc}")
        _KERNEL_CACHE = None
        return None
