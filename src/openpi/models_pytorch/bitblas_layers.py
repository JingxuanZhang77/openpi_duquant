"""BitBLAS Quantization Layers for OpenPI.

This module implements true INT4 weight quantization using Microsoft's BitBLAS library.
It integrates with DuQuant's pre-computed rotation matrices and scales for optimal performance.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

# BitBLAS imports
try:
    import bitblas
    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.quantization.utils import general_compress
    from bitblas.utils import auto_detect_nvidia_target
    BITBLAS_AVAILABLE = True
except ImportError as e:
    BITBLAS_AVAILABLE = False
    BITBLAS_IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)


@dataclass
class BitBLASConfig:
    """Configuration for BitBLAS quantization."""
    bits: int = 4
    group_size: int = 128
    enable_tuning: bool = False
    opt_M: List[int] = None  # Dynamic batch sizes, e.g., [1, 16, 32, 64]
    pack_dir: Optional[str] = None  # DuQuant pack directory
    scope: str = ""
    include_regex: str = r".*"
    exclude_regex: str = r"(?:^|\.)(norm|ln|layernorm|emb|embed|vision_tower)(?:\.|$)"
    debug: bool = False

    def __post_init__(self):
        if self.opt_M is None:
            self.opt_M = [1, 16, 32, 64]  # Default for LIBERO tasks

    @classmethod
    def from_env(cls) -> "BitBLASConfig":
        """Create config from environment variables."""
        opt_M_str = os.environ.get("OPENPI_BITBLAS_OPT_M", "1,16,32,64")
        opt_M = [int(x.strip()) for x in opt_M_str.split(",")]

        return cls(
            bits=int(os.environ.get("OPENPI_BITBLAS_WBITS", 4)),
            group_size=int(os.environ.get("OPENPI_BITBLAS_GROUP_SIZE", 128)),
            enable_tuning=os.environ.get("OPENPI_BITBLAS_ENABLE_TUNING", "0") not in ("0", "false", "False"),
            opt_M=opt_M,
            pack_dir=os.environ.get("OPENPI_BITBLAS_DUQUANT_PACKDIR", None),
            scope=os.environ.get("OPENPI_BITBLAS_SCOPE", ""),
            include_regex=os.environ.get("OPENPI_BITBLAS_INCLUDE", r".*"),
            exclude_regex=os.environ.get("OPENPI_BITBLAS_EXCLUDE",
                                        r"(?:^|\.)(norm|ln|layernorm|emb|embed|vision_tower)(?:\.|$)"),
            debug=os.environ.get("OPENPI_BITBLAS_DEBUG", "0") not in ("0", "false", "False"),
        )


class BitBLASQuantLinear(nn.Module):
    """BitBLAS W4FP16 Quantized Linear layer.

    This layer performs true INT4 weight quantization using BitBLAS optimized kernels.
    It can reuse DuQuant's pre-computed scales and rotation matrices.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        name: Layer name (for loading DuQuant pack)
        bits: Weight quantization bits (4)
        group_size: Group size for quantization (128)
        bias: Whether to include bias
        enable_tuning: Enable hardware-aware auto-tuning
        opt_M: List of batch sizes to optimize for
        duquant_packdir: Directory containing DuQuant packs (optional)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        name: str,
        bits: int = 4,
        group_size: int = 128,
        bias: bool = False,
        enable_tuning: bool = False,
        opt_M: Optional[List[int]] = None,
        duquant_packdir: Optional[str] = None,
    ):
        super().__init__()

        if not BITBLAS_AVAILABLE:
            raise RuntimeError(f"BitBLAS not available: {BITBLAS_IMPORT_ERROR}")

        # Validate dimensions for BitBLAS
        if in_features % group_size != 0:
            logger.warning(f"[BITBLAS] {name}: in_features={in_features} not divisible by group_size={group_size}")

        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.duquant_packdir = duquant_packdir

        # Will be populated by converter
        self.register_buffer("qweight", torch.empty((0,), dtype=torch.int8))
        self.register_buffer("scales", torch.empty((0,), dtype=torch.half))
        self.register_buffer("zeros", torch.empty((0,), dtype=torch.half))

        # DuQuant pack for input/output transforms
        self.duquant_pack = None

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.half))
        else:
            self.bias = None

        # BitBLAS configuration
        if opt_M is None:
            opt_M = [1, 16, 32, 64]

        # Auto-detect GPU target (e.g., "nvidia/nvidia-a40" -> sm_86)
        self.target = auto_detect_nvidia_target()

        # Create BitBLAS Matmul for W4FP16 (INT4 weight-only quantization)
        # Use TIR backend for sm_86 (Ampere) compatibility - avoids sm_90 TMA issues
        #
        # CRITICAL: We store weights as UNSIGNED INT4 [0, 15] using general_compress.
        # Original signed INT4 [-8, 7] is converted to unsigned by adding 8.
        # Therefore:
        #   - W_dtype="uint4" to match unsigned storage
        #   - with_zeros=True to apply zero-point offset (zeros=8)
        #   - zeros_mode="original" for standard dequant: W_fp16 = (W_uint - zeros) * scale
        #
        # IMPORTANT PRECISION FIX:
        #   - Use FP32 accumulator (not FP16) for numerical precision!
        #     FP16 accumulator causes ~0.87 max error vs ~0.06 with FP32.
        #   - Use per-channel scale (group_size = in_features) for best accuracy.
        #     This gives 1 scale per output channel, matching DuQuant's approach.
        effective_group_size = in_features  # Per-channel scale for accuracy
        self.effective_group_size = effective_group_size

        matmul_config = MatmulConfig(
            M=opt_M,
            N=out_features,
            K=in_features,
            A_dtype="float16",        # Activation: FP16
            W_dtype="uint4",          # Weight: UINT4 (unsigned [0,15])
            out_dtype="float16",      # Output: FP16
            accum_dtype="float32",    # Accumulator: FP32 (critical for precision!)
            with_scaling=True,        # Use per-channel scales
            with_zeros=True,          # Use zero-point offset (zeros=8)
            zeros_mode="original",    # Dequant: W = (W_uint - zeros) * scale
            group_size=effective_group_size,  # Per-channel (= in_features)
            with_bias=bias,
            layout="nt",
        )

        try:
            logger.info(f"[BITBLAS] Creating INT4 Matmul for {name}")
            # Use TIR backend for Ampere (sm_86) - more stable than TileLanguage
            self.bitblas_matmul = Matmul(
                matmul_config,
                target=self.target,
                backend="tir",
                enable_tuning=enable_tuning,
            )

            logger.info(f"[BITBLAS] Successfully created W4FP16 kernel for {name} (backend=tir)")
        except Exception as e:
            logger.error(f"[BITBLAS] Failed to create kernel for {name}: {e}")
            logger.warning(f"[BITBLAS] Will use dequantization fallback")
            self.bitblas_matmul = None

        self._weight_loaded = False
        self._debug = os.environ.get("OPENPI_BITBLAS_DEBUG", "0") not in ("0", "false", "False")

    def load_from_linear(self, linear: nn.Linear, duquant_pack=None):
        """Load and convert weights from a standard Linear layer.

        Uses DuQuant's pre-computed transforms and scales, then uses BitBLAS's
        transform_weight() function for proper weight packing.

        CRITICAL: Must use BitBLAS transform_weight() instead of general_compress
        because BitBLAS applies additional transformations (ladder permutation,
        LOP3 permutation, etc.) that are essential for correct computation.

        Args:
            linear: Source nn.Linear layer
            duquant_pack: Optional DuQuant PackResult for using pre-computed transforms
        """
        from .duquant_preprocess import load_pack

        try:
            # Load DuQuant pack if not provided
            if duquant_pack is None and self.duquant_packdir is not None:
                duquant_pack = load_pack(self.name, self.duquant_packdir)

            if duquant_pack is None:
                raise ValueError(f"DuQuant pack not found for {self.name}. "
                               f"Please run DuQuant preprocessing first.")

            # Store pack for input/output transforms
            self.duquant_pack = duquant_pack

            # Step 1: Get FP16 weight and apply DuQuant transforms (perm + R_in + R_out)
            W = linear.weight.detach().clone().half()  # (out_features, in_features)

            # Apply column permutation
            if duquant_pack.perm is not None:
                perm_t = torch.from_numpy(duquant_pack.perm).long().to(W.device)
                W = W[:, perm_t]

            # Apply R_in rotation (per-block)
            if duquant_pack.R_in_blocks:
                block_size = duquant_pack.meta.get("block_size", 16)
                n_blocks = self.in_features // block_size
                for b in range(n_blocks):
                    if b in duquant_pack.R_in_blocks:
                        R = torch.from_numpy(duquant_pack.R_in_blocks[b]).to(W.device, W.dtype)
                        start = b * block_size
                        end = start + block_size
                        W[:, start:end] = W[:, start:end] @ R

            # Apply R_out rotation (per-block)
            # NOTE: DuQuant applies R_out as `R @ W[block]` (left multiplication)
            # NOT `R.T @ W[block]` - this is critical for correctness!
            if duquant_pack.R_out_blocks:
                block_out_size = duquant_pack.meta.get("block_out_size", 16)
                n_out_blocks = self.out_features // block_out_size
                for b in range(n_out_blocks):
                    if b in duquant_pack.R_out_blocks:
                        R = torch.from_numpy(duquant_pack.R_out_blocks[b]).to(W.device, W.dtype)
                        start = b * block_out_size
                        end = start + block_out_size
                        W[start:end, :] = R @ W[start:end, :]

            # Step 2: Compute per-channel scales using MSE optimization (like DuQuant)
            # IMPORTANT: DuQuant computes scales AFTER transforms using compute_mse_scales,
            # not the pre-computed weight_scale from the pack file
            from .duquant_preprocess import compute_mse_scales

            with torch.no_grad():
                weight_scale = compute_mse_scales(W, self.bits)  # (out_features,)

            # Signed INT4 range: [-8, 7]
            max_q = 7
            min_q = -8

            # Quantize: q = clamp(round(W / scale), -8, 7)
            W_scaled = W / weight_scale[:, None]  # (out_features, in_features)
            W_int = torch.clamp(torch.round(W_scaled), min_q, max_q).to(torch.int8)

            # Step 3: Convert to unsigned INT4 [0, 15] for BitBLAS uint4 format
            W_uint = (W_int + 8).to(torch.uint8)  # [-8,7] -> [0,15]

            # Step 4: Prepare scales and zeros for BitBLAS
            # Use per-channel scale (1 group = entire input dimension)
            n_groups = 1  # Per-channel = 1 group
            scales = weight_scale[:, None].contiguous()  # (out_features, 1)
            zeros = torch.full((self.out_features, n_groups), 8.0, dtype=torch.float16)

            device = linear.weight.device

            # Step 5: Use BitBLAS transform_weight() for proper packing
            # CRITICAL: BitBLAS applies ladder permutation, LOP3 permutation, etc.
            # that are essential for correct kernel computation
            if self.bitblas_matmul is not None:
                # Move to CUDA for transform_weight
                W_uint_cuda = W_uint.to(device).to(torch.int8)  # BitBLAS expects int8 container
                scales_cuda = scales.to(device)
                zeros_cuda = zeros.to(device)

                # Use BitBLAS transform_weight
                transformed = self.bitblas_matmul.transform_weight(
                    W_uint_cuda,
                    scale=scales_cuda,
                    zeros=zeros_cuda
                )

                # Parse result
                if isinstance(transformed, list):
                    qweight = transformed[0]
                else:
                    qweight = transformed

                self.qweight = qweight.contiguous()
                self.scales = scales_cuda.contiguous()
                self.zeros = zeros_cuda.contiguous()

                if self._debug:
                    logger.info(f"[BITBLAS] {self.name}: Used BitBLAS transform_weight()")
            else:
                # Fallback: manual packing with general_compress
                W_uint_np = W_uint.cpu().numpy()
                qweight_np = general_compress(W_uint_np, source_bits=self.bits, storage_dtype=np.int8)
                qweight = torch.from_numpy(qweight_np).contiguous()

                self.qweight = qweight.to(device).contiguous()
                self.scales = scales.to(device).contiguous()
                self.zeros = zeros.to(device).contiguous()

                if self._debug:
                    logger.info(f"[BITBLAS] {self.name}: Used general_compress (fallback)")

            if self.bias is not None and linear.bias is not None:
                self.bias.copy_(linear.bias.to(torch.half))

            self._weight_loaded = True

            if self._debug:
                logger.info(f"[BITBLAS] {self.name}: Loaded weights")
                logger.info(f"  qweight shape: {self.qweight.shape}, dtype: {self.qweight.dtype}")
                logger.info(f"  scales shape: {self.scales.shape}, range: [{self.scales.min():.6f}, {self.scales.max():.6f}]")
                logger.info(f"  zeros shape: {self.zeros.shape}, value: {self.zeros[0,0]:.1f}")
                logger.info(f"  DuQuant pack: perm={duquant_pack.perm is not None}, "
                          f"R_in={len(duquant_pack.R_in_blocks) if duquant_pack.R_in_blocks else 0}, "
                          f"R_out={len(duquant_pack.R_out_blocks) if duquant_pack.R_out_blocks else 0}")

        except Exception as e:
            logger.error(f"[BITBLAS] Failed to convert {self.name}: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using BitBLAS INT4 kernel or fallback to dequantization.

        CRITICAL: Apply DuQuant input/output transforms to match quantized weights!

        Args:
            x: Input tensor (..., in_features) in FP16

        Returns:
            output: Output tensor (..., out_features) in FP16
        """
        _debug = os.environ.get("OPENPI_BITBLAS_DEBUG", "0") not in ("0", "false", "False")

        # Step 1: Apply DuQuant input transform (perm + R_in rotation)
        if self.duquant_pack is not None:
            from .duquant_preprocess import apply_input_transform_optimized
            x_transformed = apply_input_transform_optimized(
                x, self.duquant_pack,
                perm_cache=None,
                R_in_cache=None,
                block_size=self.duquant_pack.meta.get("block_size", 16)
            )
            if _debug:
                logger.debug(f"[BITBLAS][FWD] {self.name}: Input transform applied")
                logger.debug(f"  x range: [{x.min():.6f}, {x.max():.6f}]")
                logger.debug(f"  x_transformed range: [{x_transformed.min():.6f}, {x_transformed.max():.6f}]")
        else:
            x_transformed = x

        # Step 2: Matmul with quantized weights using BitBLAS kernel
        _force_fallback = os.environ.get("OPENPI_BITBLAS_FORCE_FALLBACK", "0") not in ("0", "false", "False")

        if self.bitblas_matmul is not None and not _force_fallback:
            # Use BitBLAS optimized UINT4 x FP16 kernel
            # API: bitblas_matmul(A, qweight, scales, zeros, [bias], output)
            # Note: with_zeros=True, so we pass zeros tensor for proper dequantization
            output_shape = x_transformed.shape[:-1] + (self.out_features,)
            output = torch.empty(output_shape, dtype=x_transformed.dtype, device=x_transformed.device)

            try:
                # Call BitBLAS kernel
                # CRITICAL: Must use output= keyword argument, NOT positional!
                # BitBLAS only writes to output tensor when using keyword argument.
                if self.bias is not None:
                    self.bitblas_matmul(x_transformed, self.qweight, self.scales, self.zeros, self.bias, output=output)
                else:
                    self.bitblas_matmul(x_transformed, self.qweight, self.scales, self.zeros, output=output)
                y_transformed = output

                if _debug:
                    logger.debug(f"[BITBLAS][FWD] {self.name}: BitBLAS kernel executed")
                    logger.debug(f"  y_transformed range: [{y_transformed.min():.6f}, {y_transformed.max():.6f}]")

            except Exception as e:
                logger.warning(f"[BITBLAS] Kernel execution failed for {self.name}: {e}")
                logger.warning(f"[BITBLAS] Falling back to dequantization")
                W_fp16 = self._dequantize_weight()
                y_transformed = torch.nn.functional.linear(x_transformed, W_fp16, None)
                if self.bias is not None:
                    y_transformed = y_transformed + self.bias
                if _debug:
                    logger.debug(f"[BITBLAS][FWD] {self.name}: Using dequant fallback (error)")
        else:
            # Fallback: Dequantize INT4 weights to FP16
            W_fp16 = self._dequantize_weight()
            y_transformed = torch.nn.functional.linear(x_transformed, W_fp16, None)
            if self.bias is not None:
                y_transformed = y_transformed + self.bias
            if _debug:
                logger.debug(f"[BITBLAS][FWD] {self.name}: Using dequant fallback (forced)")
                logger.debug(f"  y_transformed range: [{y_transformed.min():.6f}, {y_transformed.max():.6f}]")

        # Step 3: Apply DuQuant output restore (R_out inverse rotation)
        row_rot_mode = os.environ.get("OPENPI_DUQUANT_ROW_ROT", "restore")

        if self.duquant_pack is not None and self.duquant_pack.R_out_blocks and row_rot_mode == "restore":
            from .duquant_preprocess import apply_output_restore_optimized
            output = apply_output_restore_optimized(
                y_transformed, self.duquant_pack,
                R_out_cache=None,
                block_out_size=self.duquant_pack.meta.get("block_out_size", 16)
            )
            if _debug:
                logger.debug(f"[BITBLAS][FWD] {self.name}: Output restore applied")
                logger.debug(f"  output range: [{output.min():.6f}, {output.max():.6f}]")
        else:
            output = y_transformed
            if _debug:
                logger.debug(f"[BITBLAS][FWD] {self.name}: No output restore (row_rot_mode={row_rot_mode})")

        return output

    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize INT4 weights to FP16 for fallback forward pass.

        Uses asymmetric quantization formula (with zeros offset):
        W_fp16 = (W_uint - zeros) * scales

        Note: We use per-channel scale (1 scale per output row), so
        scales shape is (out_features, 1) and zeros shape is (out_features, 1).
        """
        # Unpack unsigned INT4 from general_compress format
        # general_compress packs 2 UINT4 values per INT8 byte:
        #   packed = (val0 << 0) | (val1 << 4)
        # where val0 is at index 2*i, val1 is at index 2*i+1

        # Use unsigned view for correct bit operations
        qweight_np = self.qweight.cpu().numpy().view(np.uint8)
        out_features, packed_in = qweight_np.shape
        in_features = packed_in * 2  # 2 INT4 per INT8

        # Unpack: each byte contains 2 unsigned INT4 values [0, 15]
        W_uint_np = np.zeros((out_features, in_features), dtype=np.uint8)
        for i in range(packed_in):
            # Low 4 bits (val0 at index 2*i)
            W_uint_np[:, 2*i] = qweight_np[:, i] & 0x0F
            # High 4 bits (val1 at index 2*i+1)
            W_uint_np[:, 2*i + 1] = (qweight_np[:, i] >> 4) & 0x0F

        W_uint = torch.from_numpy(W_uint_np).to(self.qweight.device)

        # Dequantize: W_fp16 = (W_uint - zeros) * scales
        # Per-channel: scales is (out_features, 1), zeros is (out_features, 1) = 8
        # Simple broadcast: (W_uint - 8) * scale
        scale = self.scales  # (out_features, 1)
        zero = self.zeros    # (out_features, 1) = 8

        W_fp16 = (W_uint.half() - zero) * scale

        return W_fp16


def select_targets_for_bitblas(
    model: nn.Module,
    include_regex: str,
    exclude_regex: str,
    scope_prefix: str = "",
) -> List[Tuple[str, nn.Linear]]:
    """Select Linear layers to quantize with BitBLAS.

    Reuses DuQuant's layer selection logic for consistency.
    """
    from .duquant_layers import select_targets

    return select_targets(
        model,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
        scope_prefix=scope_prefix
    )


def wrap_bitblas(
    model: nn.Module,
    layer_names: List[str],
    config: BitBLASConfig,
) -> int:
    """Replace selected Linear layers with BitBLASQuantLinear.

    Args:
        model: PyTorch model
        layer_names: List of layer names to replace
        config: BitBLAS configuration

    Returns:
        Number of layers replaced
    """
    replaced_count = 0

    for name in layer_names:
        # Navigate to parent module
        *parent_path, attr_name = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)

        original_linear = getattr(parent, attr_name)

        if not isinstance(original_linear, nn.Linear):
            logger.warning(f"[BITBLAS] {name} is not nn.Linear, skipping")
            continue

        # Create BitBLAS layer
        bitblas_layer = BitBLASQuantLinear(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            name=name,
            bits=config.bits,
            group_size=config.group_size,
            bias=original_linear.bias is not None,
            enable_tuning=config.enable_tuning,
            opt_M=config.opt_M,
            duquant_packdir=config.pack_dir,
        )

        # Convert weights
        try:
            bitblas_layer.load_from_linear(original_linear)
        except Exception as e:
            logger.error(f"[BITBLAS] Failed to convert {name}: {e}")
            continue

        # Replace module
        setattr(parent, attr_name, bitblas_layer)
        replaced_count += 1

        if config.debug:
            logger.info(f"[BITBLAS] Replaced {name} ({original_linear.in_features} -> {original_linear.out_features})")

    return replaced_count


def enable_bitblas_if_configured(model: nn.Module) -> None:
    """Entry point to enable BitBLAS quantization based on environment variables.

    Environment variables:
        OPENPI_BITBLAS_ENABLE: Set to 1 to enable
        OPENPI_BITBLAS_WBITS: Weight bits (default: 4)
        OPENPI_BITBLAS_GROUP_SIZE: Group size (default: 128)
        OPENPI_BITBLAS_ENABLE_TUNING: Enable auto-tuning (default: 0)
        OPENPI_BITBLAS_OPT_M: Batch sizes to optimize for (default: "1,16,32,64")
        OPENPI_BITBLAS_DUQUANT_PACKDIR: DuQuant pack directory
        OPENPI_BITBLAS_INCLUDE: Regex for layers to include
        OPENPI_BITBLAS_EXCLUDE: Regex for layers to exclude
        OPENPI_BITBLAS_SCOPE: Scope prefix filter
    """
    if not BITBLAS_AVAILABLE:
        logger.warning(f"[BITBLAS] Not available: {BITBLAS_IMPORT_ERROR}")
        return

    if os.environ.get("OPENPI_BITBLAS_ENABLE", "0") in ("0", "false", "False"):
        logger.info("[BITBLAS] Not enabled (OPENPI_BITBLAS_ENABLE=0)")
        return

    # Note: OPENPI_BITBLAS_FORCE_FALLBACK only affects forward() behavior,
    # not layer replacement. This allows testing INT4 storage with FP16 compute.
    force_fallback = os.environ.get("OPENPI_BITBLAS_FORCE_FALLBACK", "0") not in ("0", "false", "False")
    if force_fallback:
        logger.info("[BITBLAS] Fallback mode enabled - will use dequant in forward()")

    config = BitBLASConfig.from_env()

    logger.info("[BITBLAS] Initializing quantization...")
    logger.info(f"  Bits: W{config.bits}FP16")
    logger.info(f"  Group size: {config.group_size}")
    logger.info(f"  Opt M: {config.opt_M}")
    logger.info(f"  DuQuant pack dir: {config.pack_dir}")
    logger.info(f"  Include: {config.include_regex}")
    logger.info(f"  Exclude: {config.exclude_regex}")

    # Select target layers
    targets = select_targets_for_bitblas(
        model,
        include_regex=config.include_regex,
        exclude_regex=config.exclude_regex,
        scope_prefix=config.scope,
    )

    layer_names = [name for name, _ in targets]
    logger.info(f"[BITBLAS] Selected {len(layer_names)} layers for quantization")

    if config.debug:
        for name in layer_names[:10]:
            logger.info(f"  - {name}")
        if len(layer_names) > 10:
            logger.info(f"  ... and {len(layer_names) - 10} more")

    # Replace layers
    replaced_count = wrap_bitblas(model, layer_names, config)

    logger.info(f"[BITBLAS] Successfully replaced {replaced_count}/{len(layer_names)} layers")

    if replaced_count == 0:
        logger.warning("[BITBLAS] No layers were replaced!")
