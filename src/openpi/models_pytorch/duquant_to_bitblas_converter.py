"""DuQuant to BitBLAS Weight Converter.

This module converts DuQuant fake-quantized weights to true INT4 format for BitBLAS.
It reuses DuQuant's pre-computed rotation matrices and scales.
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from .duquant_preprocess import PackResult, load_pack

logger = logging.getLogger(__name__)


class DuQuantToBitBLASConverter:
    """Convert DuQuant fake-quantized weights to BitBLAS INT4 format.

    This converter:
    1. Loads DuQuant's pre-computed rotation matrices and scales
    2. Applies transformations (permutation + rotations) to weights
    3. Quantizes to INT4 using DuQuant's scales
    4. Packs INT4 values into INT8 storage (2 values per byte)
    """

    def __init__(self, bits: int = 4, group_size: int = 128, debug: bool = False):
        """Initialize converter.

        Args:
            bits: Target quantization bits (4)
            group_size: Group size for quantization (128)
            debug: Enable debug logging
        """
        self.bits = bits
        self.group_size = group_size
        self.debug = debug

        if bits != 4:
            raise ValueError(f"Only 4-bit quantization supported, got {bits}")

    def convert_layer(
        self,
        layer_name: str,
        original_linear: nn.Linear,
        duquant_packdir: Optional[str],
        duquant_pack: Optional[PackResult] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a single Linear layer to BitBLAS INT4 format.

        Args:
            layer_name: Name of the layer
            original_linear: Source nn.Linear layer
            duquant_packdir: Directory containing DuQuant packs
            duquant_pack: Pre-loaded DuQuant pack (optional)

        Returns:
            qweight: INT8 tensor storing packed INT4 weights (out_features, in_features//2)
            scales: FP16 scales (out_features, in_features//group_size)
            zeros: FP16 zeros (out_features, in_features//group_size) - all zeros for symmetric quant
        """
        # Load DuQuant pack
        if duquant_pack is None:
            if duquant_packdir is None:
                raise ValueError(f"Either duquant_packdir or duquant_pack must be provided for {layer_name}")

            duquant_pack = load_pack(layer_name, duquant_packdir)

        if duquant_pack is None:
            raise ValueError(f"DuQuant pack not found for {layer_name} in {duquant_packdir}")

        # Get original FP16/BF16 weights
        W_original = original_linear.weight.data.float()  # (out_features, in_features)
        out_features, in_features = W_original.shape

        if self.debug:
            logger.info(f"[CONVERTER] {layer_name}")
            logger.info(f"  Weight shape: {W_original.shape}")
            logger.info(f"  Weight range: [{W_original.min():.6f}, {W_original.max():.6f}]")

        # Step 1: Apply DuQuant transformations
        W_transformed = self._apply_duquant_transforms(W_original, duquant_pack)

        if self.debug:
            logger.info(f"  Transformed weight range: [{W_transformed.min():.6f}, {W_transformed.max():.6f}]")

        # Step 2: Quantize to INT4 using DuQuant scales
        qweight_int4, scales, zeros = self._quantize_to_int4(
            W_transformed,
            duquant_pack.weight_scale,
            in_features,
            out_features,
        )

        if self.debug:
            logger.info(f"  INT4 range: [{qweight_int4.min()}, {qweight_int4.max()}]")
            logger.info(f"  Scales shape: {scales.shape}, range: [{scales.min():.6f}, {scales.max():.6f}]")

        # Step 3: Pack INT4 to INT8 storage
        qweight_packed = self._pack_int4_to_int8(qweight_int4)

        if self.debug:
            logger.info(f"  Packed shape: {qweight_packed.shape}")
            logger.info(f"  Memory saved: {W_original.element_size() * W_original.nelement() / (1024**2):.2f} MB -> {qweight_packed.element_size() * qweight_packed.nelement() / (1024**2):.2f} MB")

        return qweight_packed, scales, zeros

    def _apply_duquant_transforms(self, W: torch.Tensor, pack: PackResult) -> torch.Tensor:
        """Apply DuQuant transformations: permutation + input rotation + output rotation.

        Args:
            W: Original weights (out_features, in_features)
            pack: DuQuant PackResult

        Returns:
            W_transformed: Transformed weights
        """
        W_t = W.clone()
        device = W.device
        dtype = W.dtype

        # Step 1: Apply column permutation
        if pack.perm is not None:
            perm_t = torch.from_numpy(pack.perm).to(device).long()
            W_t = W_t.index_select(dim=1, index=perm_t)

            if self.debug:
                logger.debug(f"    Applied permutation: {len(pack.perm)} columns")

        # Step 2: Apply input block rotations (R_in)
        if pack.R_in_blocks:
            in_features = W_t.shape[1]
            block_size = pack.meta.get("block_size", 16)
            n_blocks = (in_features + block_size - 1) // block_size

            W_t2 = W_t.clone()
            rotated_blocks = 0

            for b in range(n_blocks):
                if b not in pack.R_in_blocks:
                    continue

                start = b * block_size
                end = min((b + 1) * block_size, in_features)
                actual_size = end - start

                R = torch.from_numpy(pack.R_in_blocks[b][:actual_size, :actual_size]).to(device, dtype)
                W_t2[:, start:end] = W_t[:, start:end] @ R
                rotated_blocks += 1

            W_t = W_t2

            if self.debug:
                logger.debug(f"    Applied input rotation: {rotated_blocks}/{n_blocks} blocks")

        # Step 3: Apply output block rotations (R_out)
        row_rot_mode = pack.meta.get("row_rot_mode", "0")
        if pack.R_out_blocks and row_rot_mode != "0":
            out_features = W_t.shape[0]
            block_out_size = pack.meta.get("block_out_size", block_size)
            n_row_blocks = (out_features + block_out_size - 1) // block_out_size

            W_t2 = W_t.clone()
            rotated_blocks = 0

            for b in range(n_row_blocks):
                if b not in pack.R_out_blocks:
                    continue

                rs = b * block_out_size
                re = min((b + 1) * block_out_size, out_features)
                actual_size = re - rs

                Rb = torch.from_numpy(pack.R_out_blocks[b][:actual_size, :actual_size]).to(device, dtype)
                W_t2[rs:re, :] = Rb @ W_t[rs:re, :]
                rotated_blocks += 1

            W_t = W_t2

            if self.debug:
                logger.debug(f"    Applied output rotation: {rotated_blocks}/{n_row_blocks} blocks (mode={row_rot_mode})")

        return W_t

    def _quantize_to_int4(
        self,
        W: torch.Tensor,
        duquant_scales: np.ndarray,
        in_features: int,
        out_features: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize transformed weights to INT4.

        DuQuant uses per-channel (per-output) scales, but we need per-group scales for BitBLAS.
        Strategy: Repeat per-channel scales across groups.

        Args:
            W: Transformed weights (out_features, in_features)
            duquant_scales: DuQuant per-channel scales (out_features,)
            in_features: Input features
            out_features: Output features

        Returns:
            W_int4: INT4 weights (out_features, in_features)
            scales: FP16 per-group scales (out_features, n_groups)
            zeros: FP16 zeros (out_features, n_groups) - all zeros
        """
        device = W.device
        n_groups = in_features // self.group_size

        # Convert DuQuant per-channel scales to per-group format
        # DuQuant scale is per-output-channel, so we repeat it for each group
        scales_np = duquant_scales  # (out_features,)
        scales = torch.from_numpy(scales_np).to(device, dtype=torch.half)
        scales = scales[:, None].expand(out_features, n_groups)  # (out_features, n_groups)

        # Symmetric quantization: zeros are all 0
        zeros = torch.zeros_like(scales)

        # Quantize per group
        # DuQuant uses symmetric quantization: signed INT4 [-8, 7]
        max_q = 2 ** (self.bits - 1) - 1  # 7
        min_q = -(2 ** (self.bits - 1))   # -8

        W_int4 = torch.zeros_like(W, dtype=torch.int8)

        for g in range(n_groups):
            start_idx = g * self.group_size
            end_idx = min(start_idx + self.group_size, in_features)

            # Get scale for this group (same for all groups from DuQuant)
            scale = scales[:, g:g+1]  # (out_features, 1)

            # Quantize: q = round(W / scale)
            W_group = W[:, start_idx:end_idx]
            W_scaled = W_group / scale
            W_quantized = torch.clamp(torch.round(W_scaled), min_q, max_q).to(torch.int8)

            W_int4[:, start_idx:end_idx] = W_quantized

        # Keep as signed INT4 [-8, 7] - BitBLAS will handle the format internally
        return W_int4, scales, zeros

    def _pack_int4_to_int8(self, qweight_int4: torch.Tensor) -> torch.Tensor:
        """Pack two INT4 values into one INT8 byte.

        Args:
            qweight_int4: INT4 weights as INT8 tensor (out_features, in_features)

        Returns:
            qweight_packed: Packed INT8 tensor (out_features, in_features//2)
        """
        out_features, in_features = qweight_int4.shape

        # Ensure in_features is even
        if in_features % 2 != 0:
            raise ValueError(f"in_features must be even for packing, got {in_features}")

        # Pack two 4-bit values into one byte
        # Low 4 bits: first value, High 4 bits: second value
        qweight_packed = torch.zeros(
            (out_features, in_features // 2),
            dtype=torch.int8,
            device=qweight_int4.device
        )

        # Convert to numpy for bit operations
        qweight_np = qweight_int4.cpu().numpy()
        packed_np = np.zeros((out_features, in_features // 2), dtype=np.int8)

        for i in range(in_features // 2):
            # Get two consecutive int4 values
            val1 = qweight_np[:, 2*i] & 0x0F      # Low 4 bits
            val2 = qweight_np[:, 2*i + 1] & 0x0F  # Low 4 bits

            # Pack: val2 in high 4 bits, val1 in low 4 bits
            packed_np[:, i] = (val2 << 4) | val1

        qweight_packed = torch.from_numpy(packed_np).to(qweight_int4.device)

        return qweight_packed


def unpack_int4_from_int8(qweight_packed: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Unpack INT8 storage back to INT4 values.

    Args:
        qweight_packed: Packed INT8 tensor (out_features, in_features//2)
        out_features: Output features
        in_features: Input features

    Returns:
        qweight_int4: Unpacked INT4 as INT8 tensor (out_features, in_features)
    """
    qweight_int4 = torch.zeros(
        (out_features, in_features),
        dtype=torch.int8,
        device=qweight_packed.device
    )

    # Convert to numpy for bit operations
    packed_np = qweight_packed.cpu().numpy()
    unpacked_np = np.zeros((out_features, in_features), dtype=np.int8)

    for i in range(in_features // 2):
        packed_byte = packed_np[:, i]

        # Unpack
        val1 = packed_byte & 0x0F            # Low 4 bits
        val2 = (packed_byte >> 4) & 0x0F     # High 4 bits

        # Sign extend from 4-bit to 8-bit for signed INT4 [-8, 7]
        val1 = np.where(val1 > 7, val1 - 16, val1).astype(np.int8)
        val2 = np.where(val2 > 7, val2 - 16, val2).astype(np.int8)

        unpacked_np[:, 2*i] = val1
        unpacked_np[:, 2*i + 1] = val2

    qweight_int4 = torch.from_numpy(unpacked_np).to(qweight_packed.device)

    return qweight_int4
