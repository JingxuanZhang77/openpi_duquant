from __future__ import annotations

import logging
import math
import os
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
from torch import nn
import torch.nn.functional as F

from ..duquant_preprocess import (
    PackResult,
    PercentileCalibrator,
    apply_bias_row_rot_optimized,
    qmax,
    transform_weight_for_forward_optimized,
)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..duquant_layers import DuQuantConfig


def _ceil_to(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def pack_int4_nibbles(q_w: torch.Tensor, in_align: int, out_align: int) -> torch.Tensor:
    """
    q_w: [out, in] int8 with values in [-8, 7].
    对齐到 in_align/out_align，做 0-padding；再把相邻两个 4bit pack 成 1 字节。
    返回 packed: [out_aligned, in_aligned // 2] uint8。
    """
    if q_w.dtype != torch.int8:
        raise ValueError("q_w must be int8 with values in [-8, 7]")
    out_features, in_features = q_w.shape
    out_aligned = _ceil_to(out_features, out_align)
    in_aligned = _ceil_to(in_features, in_align)
    if in_aligned % 2 != 0:
        raise ValueError("Aligned input features must be even for nibble packing")

    device = q_w.device
    padded = torch.zeros((out_aligned, in_aligned), dtype=torch.int8, device=device)
    padded[:out_features, :in_features] = q_w
    q_mod = torch.remainder(padded.to(torch.int16), 16).to(torch.uint8)
    low = q_mod[:, 0::2]
    high = q_mod[:, 1::2]
    packed = low | (high << 4)
    return packed.contiguous()


class QLinearW4A8BitBLAS(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        name: str,
        cfg: "DuQuantConfig",
        pack: PackResult,
        *,
        weight_bits: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.cfg = cfg

        self.weight_bits = 4 if weight_bits is None else int(weight_bits)
        if self.weight_bits != 4:
            raise ValueError(f"QLinearW4A8BitBLAS currently supports 4-bit weights, got {self.weight_bits}")

        self.act_bits = cfg.act_bits
        self.calibrator = (
            PercentileCalibrator(percentile=cfg.act_percentile, max_batches=cfg.calib_batches)
            if self.act_bits > 0 else None
        )
        self._act_scale_cache: Dict[Tuple[str, torch.dtype], torch.Tensor] = {}

        self.in_align = 64
        self.out_align = 64
        self.in_features_aligned = _ceil_to(self.in_features, self.in_align)
        self.out_features_aligned = _ceil_to(self.out_features, self.out_align)

        # Cache transform tensors on matching device/dtype
        weight = base.weight.detach()
        dtype = weight.dtype
        device = weight.device

        perm_cache = torch.from_numpy(pack.perm).long().to(device=device) if pack.perm is not None else None
        R_in_cache: Dict[int, torch.Tensor] = {}
        if pack.R_in_blocks:
            for idx, R in pack.R_in_blocks.items():
                R_in_cache[idx] = torch.from_numpy(R).to(device=device, dtype=dtype)
        R_out_cache: Dict[int, torch.Tensor] = {}
        if pack.R_out_blocks:
            for idx, R in pack.R_out_blocks.items():
                R_out_cache[idx] = torch.from_numpy(R).to(device=device, dtype=dtype)

        block_size = int(pack.meta.get("block_size", cfg.block_size))
        block_out_size = int(pack.meta.get("block_out_size", cfg.block_out_size))
        apply_row_rot = (cfg.row_rot_mode != "0")

        W_t, s_w = transform_weight_for_forward_optimized(
            weight,
            pack,
            weight_bits=self.weight_bits,
            apply_row_rot=apply_row_rot,
            perm_cache=perm_cache,
            R_in_cache=R_in_cache,
            R_out_cache=R_out_cache,
            block_size=block_size,
            block_out_size=block_out_size,
        )

        self.register_buffer("s_w", self._pad_scales(s_w, self.out_features_aligned).to(dtype=torch.float16))

        q_w = torch.round(
            W_t.to(torch.float32) / self.s_w[: self.out_features].to(torch.float32)[:, None]
        )
        q_w = torch.clamp(q_w, -8, 7).to(torch.int8)
        if os.environ.get("OPENPI_DUQUANT_DEBUG_QW", "0") == "1":
            print(f"[BITBLAS][DEBUG] {self.name} quantized weight shape {tuple(q_w.shape)}")
            print(f"[BITBLAS][DEBUG] {self.name} base weight shape {tuple(W_t.shape)}")

        packed = pack_int4_nibbles(q_w, self.in_align, self.out_align)
        self.register_buffer("packed_w", packed)

        if base.bias is not None:
            bias_vec = base.bias.detach().clone()
            if apply_row_rot and pack.R_out_blocks:
                bias_vec = apply_bias_row_rot_optimized(bias_vec, pack, R_out_cache, block_out_size)
            self.bias = nn.Parameter(bias_vec)
        else:
            self.bias = None

        self._block_size = block_size
        self._block_out_size = block_out_size

        self.pack_meta = dict(pack.meta)
        self.register_buffer(
            "input_mask",
            torch.tensor(
                [1.0] * self.in_features + [0.0] * (self.in_features_aligned - self.in_features),
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self._debug_logged = False

        logging.info(
            "[BITBLAS][CACHE] name=%s in=%d->%d out=%d->%d pack(block=%d, block_out=%d, perm=%s)",
            self.name,
            self.in_features,
            self.in_features_aligned,
            self.out_features,
            self.out_features_aligned,
            block_size,
            block_out_size,
            "yes" if pack.perm is not None else "no",
        )

    def extra_repr(self) -> str:
        return (
            f"name={self.name}, in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_bits={self.weight_bits}, act_bits={self.act_bits}"
        )

    def _pad_scales(self, scales: torch.Tensor, target: int) -> torch.Tensor:
        if scales.ndim != 1:
            raise ValueError("scales must be 1-D")
        if scales.shape[0] == target:
            return scales.clone()
        pad = target - scales.shape[0]
        if pad < 0:
            return scales[:target].clone()
        padded = torch.ones(target, dtype=scales.dtype, device=scales.device)
        padded[: scales.shape[0]] = scales
        return padded

    def _get_act_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_bits <= 0:
            return torch.ones(
                self.in_features_aligned, dtype=torch.float16, device=x.device
            )

        cache_key = (str(x.device), x.dtype)
        if self.calibrator is not None and not self.calibrator.is_full():
            self.calibrator.observe(x)
            self._act_scale_cache.pop(cache_key, None)

        cached = self._act_scale_cache.get(cache_key)
        if cached is not None:
            return cached

        scale_vec: Optional[torch.Tensor] = None
        if self.calibrator is not None and self.calibrator.is_full():
            scale_vec = self.calibrator.finalize()

        if scale_vec is None or scale_vec.numel() != self.in_features:
            with torch.no_grad():
                x_abs = torch.abs(x.detach().to(torch.float32, non_blocking=True))
                C = x_abs.shape[-1]
                x2d = x_abs.reshape(-1, C)
                q = torch.quantile(
                    x2d, self.cfg.act_percentile / 100.0, dim=0
                )
                scale_vec = torch.clamp(q, min=1e-6)

        scale_vec = scale_vec.to(device=x.device, dtype=torch.float32)
        scale_vec = scale_vec / qmax(self.act_bits)
        scale_vec = torch.clamp(scale_vec, min=1e-6)
        scale_vec = self._pad_scales(scale_vec, self.in_features_aligned)
        scale_vec = scale_vec.to(torch.float16)
        self._act_scale_cache[cache_key] = scale_vec
        return scale_vec

    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        original_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        needs_pad = self.in_features_aligned > self.in_features
        if needs_pad:
            pad = self.in_features_aligned - self.in_features
            x_2d = F.pad(x_2d, (0, pad), value=0.0)
        return x_2d, needs_pad

    def _prepare_bias(self, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        bias_vec = self.bias.to(device=device, dtype=dtype)
        if self.out_features_aligned == self.out_features:
            return bias_vec
        pad = self.out_features_aligned - self.out_features
        return F.pad(bias_vec, (0, pad), value=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in, padded = self._prepare_input(x)
        s_a = self._get_act_scale(x)
        if s_a.device != x_in.device:
            s_a = s_a.to(device=x_in.device)

        try:
            import bitblas  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime failure path
            raise RuntimeError(
                "BitBLAS is not available, set OPENPI_DUQUANT_BACKEND=fake to fallback."
            ) from exc

        linear_fn = getattr(bitblas, "linear_w4a8", None)
        if linear_fn is None:
            ops = getattr(bitblas, "ops", None)
            linear_fn = getattr(ops, "linear_w4a8", None) if ops is not None else None
        if linear_fn is None:
            raise RuntimeError("bitblas.linear_w4a8 kernel not found in BitBLAS installation")

        bias = self._prepare_bias(x_in.device, x_in.dtype)
        try:
            y = linear_fn(
                x_in,
                self.packed_w.to(device=x_in.device),
                self.s_w.to(device=x_in.device),
                s_a,
                bias=bias,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"BitBLAS linear failed for {self.name} with packed_w.shape={tuple(self.packed_w.shape)} "
                f"(out_aligned={self.out_features_aligned}, in_aligned={self.in_features_aligned})"
            ) from exc

        if y.shape[-1] != self.out_features_aligned:
            raise RuntimeError(
                f"Unexpected BitBLAS output shape {y.shape}, expected last dim {self.out_features_aligned}"
            )

        if self.out_features_aligned != self.out_features:
            y = y[..., : self.out_features]

        y = y.reshape(*x.shape[:-1], self.out_features)

        if not self._debug_logged:
            logging.info(
                "[BITBLAS][FORWARD] %s input=%s output=%s weight_bits=%d act_bits=%d",
                self.name,
                tuple(x.shape),
                tuple(y.shape),
                self.weight_bits,
                self.act_bits,
            )
            self._debug_logged = True

        return y
