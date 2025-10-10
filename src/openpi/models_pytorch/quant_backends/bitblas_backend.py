import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def _try_import_bitblas():
    try:
        import bitblas  # type: ignore
        return bitblas
    except Exception:
        return None


@dataclass
class BitBlasConfig:
    use: bool
    a_dtype: str = "int8"   # activation dtype for BitBLAS
    w_dtype: str = "int4"   # weight dtype for BitBLAS
    out_dtype: str = "float16"
    accum_dtype: str = "float16"


class BitBlasLinearBackend:
    """Weight-int4, activation-int8 backend via BitBLAS Matmul.

    This backend assumes per-input-channel activation scale `s_a` and per-output-channel
    weight scale `s_w`. To incorporate activation scales into the int8 path, it pre-folds
    `s_a` into the weight matrix columns: W_adj[:, j] = W[:, j] * s_a[j]. Then it quantizes
    W_adj to int4 with per-row scale, packs it with BitBLAS, and runs Matmul on int8 inputs.
    """

    def __init__(self, in_features: int, out_features: int, device: torch.device) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.device = device
        self._bitblas = _try_import_bitblas()
        self._matmul = None
        self._packed_w = None
        self._packed_scale = None
        self._cached_key: Optional[Tuple[int, str]] = None  # (id_of_s_a_storage, device_str)

    @property
    def available(self) -> bool:
        return self._bitblas is not None and torch.cuda.is_available()

    def _ensure_matmul(self) -> None:
        if self._matmul is not None:
            return
        bitblas = self._bitblas
        assert bitblas is not None
        # Use dynamic M (sequence/batch) support with default opt_M
        target = os.environ.get("OPENPI_BITBLAS_TARGET") or os.environ.get("TVM_TARGET") or "cuda -arch=sm_80"
        cfg = bitblas.MatmulConfig(
            M=None,
            N=self.out_features,
            K=self.in_features,
            A_dtype="int8",          # INT8 activations
            W_dtype="int4",          # INT4 weights (signed, range [-8, 7])
            out_dtype="float16",
            accum_dtype="float16",
            layout="nt",
            with_bias=False,
            # Don't use with_scaling - we'll apply scales manually in run()
            # INT source format doesn't support with_scaling=True
            with_scaling=False,
            with_zeros=False,
            group_size=self.in_features,
        )
        # Pass explicit target and backend to avoid SM90/TMA kernels and CUTLASS compilation errors
        try:
            self._matmul = bitblas.Matmul(
                config=cfg,
                target=target,
                enable_tuning=False,
                backend="tir"  # Use TIR backend to avoid CUTLASS/TMA compilation errors
            )
        except TypeError:
            # Older bitblas signature without 'target' keyword
            self._matmul = bitblas.Matmul(config=cfg)

    def rebuild_weights(
        self,
        W_t: torch.Tensor,
        s_a: torch.Tensor,
        compute_mse_scales,
    ) -> None:
        """Pack weights for BitBLAS given transformed W and activation scales.

        W_t: [O, I] float tensor on CUDA
        s_a: [I] float tensor on CUDA
        compute_mse_scales: function to compute per-row scales for given bits
        """
        assert self.available, "BitBLAS backend not available on this host"
        self._ensure_matmul()
        # Fold activation scale into weights per input column
        # W_adj[o, j] = W_t[o, j] * s_a[j]
        W_adj = W_t * s_a.view(1, -1)
        # Per-output scales (row-wise) for 4-bit
        s_w = compute_mse_scales(W_adj, 4)
        # Quantize to INT4 range [-8, 7] in int8 container
        max_q = 7.0
        Qw = torch.clamp(torch.round(W_adj / s_w.view(-1, 1)), -max_q - 1.0, max_q).to(torch.int8)
        # Pack with BitBLAS
        packed = self._matmul.transform_weight(Qw)
        # Keep references on device
        self._packed_w = packed
        self._packed_scale = s_w.view(-1, 1).to(dtype=torch.float16)

    def maybe_rebuild(self, W_t: torch.Tensor, s_a: torch.Tensor, compute_mse_scales) -> None:
        key = (s_a.storage().data_ptr(), str(W_t.device))
        if self._cached_key == key and self._packed_w is not None:
            return
        self.rebuild_weights(W_t, s_a, compute_mse_scales)
        self._cached_key = key

    def run(self, x_t: torch.Tensor, s_a: torch.Tensor) -> torch.Tensor:
        assert self._matmul is not None and self._packed_w is not None and self._packed_scale is not None
        # Quantize activations to int8 using s_a
        Qa = torch.clamp(torch.round(x_t / s_a.view(1, -1)), -128, 127).to(torch.int8)
        # Run bitblas matmul (INT8 x INT4 -> FP16) without internal scaling
        # Returns raw integer matmul result cast to float16
        y = self._matmul(Qa, self._packed_w)
        # Manually apply weight scales (per-row)
        y = y * self._packed_scale.t()
        return y
