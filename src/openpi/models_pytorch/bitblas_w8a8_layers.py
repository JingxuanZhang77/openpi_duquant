"""BitBLAS W8A8 Quantization Layer.

True INT8×INT8 quantization using BitBLAS kernels with native Tensor Core support.
No DuQuant transforms (perm/rotation) needed - INT8 precision is sufficient.

Architecture:
    FP16 Input → INT8 quantize → INT8×INT8 kernel → INT32 accum → FP16 Output

Benefits:
    - 50% memory reduction (INT8 weights vs FP16)
    - 1.5-2x speedup (native INT8 Tensor Core on SM86)
    - Simple implementation (no transforms needed)
"""

import logging
import os
import re
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import BitBLAS
try:
    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target
    from bitblas.cache import global_operator_cache
    BITBLAS_AVAILABLE = True
except ImportError:
    BITBLAS_AVAILABLE = False
    global_operator_cache = None
    logger.warning("[W8A8] BitBLAS not available")

# Global cache for Matmul objects: (N, K, tuple(opt_M), enable_tuning) -> Matmul
_MATMUL_CACHE = {}
_TUNING_DONE = False  # Flag to track if we've done tuning in this session

# Activation capture for MSE testing
_ACTIVATION_CAPTURE = {
    "enabled": False,
    "layer_name": None,  # Which layer to capture (None = first match)
    "max_samples": 10,
    "samples": [],  # List of (x, W_fp16, weight_scale, y_bitblas) tuples
}


def enable_activation_capture(layer_name: Optional[str] = None, max_samples: int = 10):
    """Enable activation capture for MSE testing.

    Args:
        layer_name: Regex pattern for layer name to capture (None = any layer)
        max_samples: Maximum number of samples to capture
    """
    _ACTIVATION_CAPTURE["enabled"] = True
    _ACTIVATION_CAPTURE["layer_name"] = layer_name
    _ACTIVATION_CAPTURE["max_samples"] = max_samples
    _ACTIVATION_CAPTURE["samples"] = []
    logger.info(f"[W8A8] Activation capture enabled: layer={layer_name}, max_samples={max_samples}")


def disable_activation_capture():
    """Disable activation capture."""
    _ACTIVATION_CAPTURE["enabled"] = False
    logger.info("[W8A8] Activation capture disabled")


def get_captured_activations():
    """Get captured activations for MSE testing."""
    return _ACTIVATION_CAPTURE["samples"]


def save_captured_activations(path: str):
    """Save captured activations to file."""
    samples = _ACTIVATION_CAPTURE["samples"]
    if not samples:
        logger.warning("[W8A8] No activations captured")
        return

    save_data = []
    for sample in samples:
        save_data.append({
            "layer_name": sample["layer_name"],
            "x": sample["x"].cpu(),
            "W_fp16": sample["W_fp16"].cpu(),
            "weight_scale": sample["weight_scale"].cpu(),
            "y_bitblas": sample["y_bitblas"].cpu(),
        })

    torch.save(save_data, path)
    logger.info(f"[W8A8] Saved {len(save_data)} activation samples to {path}")


def save_tuning_cache():
    """Save all tuned operators to the BitBLAS database for persistence.

    NOTE: BitBLAS has a bug where save_into_database() without database_path
    creates a temp directory that gets deleted. We must explicitly pass
    the database_path to persist the cache.
    """
    global _TUNING_DONE
    if not BITBLAS_AVAILABLE or global_operator_cache is None:
        return

    try:
        from bitblas.cache import get_database_path
        from bitblas.utils import auto_detect_nvidia_target

        database_path = get_database_path()
        target = auto_detect_nvidia_target()

        cache_size = global_operator_cache.size()
        if cache_size > 0:
            # CRITICAL: Must pass database_path explicitly to avoid tempfile bug
            global_operator_cache.save_into_database(database_path=database_path, target=target)
            logger.info(f"[W8A8] Saved {cache_size} tuned operators to {database_path}")
            _TUNING_DONE = True
    except Exception as e:
        logger.warning(f"[W8A8] Failed to save tuning cache: {e}")


def load_tuning_cache():
    """Load previously tuned operators from the BitBLAS database.

    Returns number of operators loaded from cache.
    """
    if not BITBLAS_AVAILABLE or global_operator_cache is None:
        return 0

    try:
        from bitblas.cache import get_database_path
        from bitblas.utils import auto_detect_nvidia_target

        database_path = get_database_path()
        target = auto_detect_nvidia_target()

        # Check if database exists
        if not os.path.exists(database_path):
            logger.info(f"[W8A8] No cache found at {database_path}")
            return 0

        # Load from database with explicit path
        global_operator_cache.load_from_database(database_path=database_path, target=target)
        cache_size = global_operator_cache.size()
        if cache_size > 0:
            logger.info(f"[W8A8] Loaded {cache_size} tuned operators from {database_path}")
        return cache_size
    except Exception as e:
        logger.warning(f"[W8A8] Failed to load tuning cache: {e}")
    return 0


class BitBLASW8A8Linear(nn.Module):
    """INT8×INT8 Linear layer using BitBLAS.

    Weights are stored as INT8, activations are quantized to INT8 at runtime.
    Uses native INT8 Tensor Core for computation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        name: str = "",
        enable_tuning: bool = False,
        opt_M: list = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.enable_tuning = enable_tuning

        if opt_M is None:
            opt_M = [1, 16, 32, 64]

        # Weight storage (INT8)
        self.register_buffer("qweight", None)
        # Per-channel weight scale: W_fp16 = qweight * weight_scale
        self.register_buffer("weight_scale", None)
        # Per-tensor activation scale (computed dynamically or calibrated)
        self.register_buffer("act_scale", None)
        # Bias (optional)
        self.register_buffer("bias", None)
        # Fake weight property for compatibility (some code accesses .weight)
        self._weight_placeholder = None

        # BitBLAS matmul operator
        self.bitblas_matmul = None

        if not BITBLAS_AVAILABLE:
            logger.warning(f"[W8A8] {name}: BitBLAS not available, will use fallback")
            return

        try:
            # Check cache first
            cache_key = (out_features, in_features, tuple(opt_M), enable_tuning)
            if cache_key in _MATMUL_CACHE:
                self.bitblas_matmul = _MATMUL_CACHE[cache_key]
                self.has_bias = bias
                logger.info(f"[W8A8] {name}: Reusing cached Matmul for ({out_features}, {in_features})")
                return

            target = auto_detect_nvidia_target()
            logger.info(f"[W8A8] {name}: Target={target}, creating new Matmul for ({out_features}, {in_features})")

            # W8A8 MatmulConfig
            matmul_config = MatmulConfig(
                M=opt_M,
                N=out_features,
                K=in_features,
                A_dtype="int8",
                W_dtype="int8",
                out_dtype="int32",
                accum_dtype="int32",
                layout="nt",
                with_scaling=False,
                with_bias=False,
            )
            self.has_bias = bias
            self._matmul_config = matmul_config  # Store for cache persistence

            self.bitblas_matmul = Matmul(
                matmul_config,
                target=target,
                backend="tir",
                enable_tuning=enable_tuning,
            )
            # Cache the matmul object locally
            _MATMUL_CACHE[cache_key] = self.bitblas_matmul

            # CRITICAL: Also add to global_operator_cache for persistence
            # BitBLAS doesn't auto-add to cache, we must do it manually
            if enable_tuning and global_operator_cache is not None:
                global_operator_cache.add(matmul_config, self.bitblas_matmul)
                logger.info(f"[W8A8] {name}: Added tuned operator to global cache for persistence")

            logger.info(f"[W8A8] {name}: BitBLAS Matmul created and cached")

        except Exception as e:
            logger.error(f"[W8A8] {name}: Failed to create BitBLAS Matmul: {e}")
            self.bitblas_matmul = None

    def load_from_linear(self, linear: nn.Linear, act_scale: Optional[torch.Tensor] = None):
        """Load weights from a FP16 Linear layer and quantize to INT8.

        Args:
            linear: Source FP16 Linear layer
            act_scale: Optional pre-calibrated activation scale. If None, will use dynamic quantization.
        """
        W = linear.weight.data.float()  # (out_features, in_features)

        # Step 1: Compute per-channel weight scale using absmax
        # scale = max(|W|) / 127
        weight_absmax = W.abs().max(dim=1)[0]  # (out_features,)
        weight_scale = weight_absmax / 127.0
        weight_scale = torch.clamp(weight_scale, min=1e-8)  # Avoid division by zero

        # Step 2: Quantize weights to INT8 [-127, 127]
        W_scaled = W / weight_scale[:, None]
        W_int8 = torch.clamp(torch.round(W_scaled), -127, 127).to(torch.int8)

        # Step 3: Store quantized weights and scale
        device = linear.weight.device

        if self.bitblas_matmul is not None:
            # Use BitBLAS transform_weight for proper packing
            try:
                qweight = self.bitblas_matmul.transform_weight(W_int8.to(device))
                if isinstance(qweight, list):
                    qweight = qweight[0]
                self.qweight = qweight
                logger.info(f"[W8A8] {self.name}: BitBLAS transform_weight successful, shape={qweight.shape}")
            except Exception as e:
                logger.warning(f"[W8A8] {self.name}: transform_weight failed: {e}, using raw INT8")
                self.qweight = W_int8.to(device)
        else:
            self.qweight = W_int8.to(device)

        self.weight_scale = weight_scale.half().to(device)

        # Step 4: Set activation scale
        if act_scale is not None:
            self.act_scale = act_scale.to(device)
        else:
            # Will use dynamic quantization
            self.act_scale = None

        # Step 5: Handle bias
        if linear.bias is not None:
            self.bias = linear.bias.data.clone().to(device)

        logger.info(f"[W8A8] {self.name}: Loaded, qweight={self.qweight.shape}, "
                   f"weight_scale={self.weight_scale.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with INT8×INT8 computation via BitBLAS.

        Args:
            x: Input tensor of shape (..., in_features), any dtype

        Returns:
            Output tensor of shape (..., out_features), same dtype as input
        """
        original_shape = x.shape
        original_dtype = x.dtype

        # Flatten to 2D
        x_2d = x.view(-1, self.in_features)
        batch_size = x_2d.shape[0]

        # Step 1: Optimized dynamic quantization to INT8
        # Stay in FP16 as long as possible, then convert to int8 directly
        act_absmax = x_2d.abs().max()
        # Use inverse scale for faster multiplication
        scale_inv = 127.0 / (act_absmax + 1e-8)
        x_int8 = (x_2d * scale_inv).round().clamp(-127, 127).to(torch.int8)
        act_scale = act_absmax.float() / 127.0

        # Step 2: INT8×INT8 matmul via BitBLAS
        if self.bitblas_matmul is not None:
            output_int32 = torch.empty(batch_size, self.out_features, dtype=torch.int32, device=x.device)
            self.bitblas_matmul(x_int8, self.qweight, output=output_int32)

            # Step 3: Dequantize: y = y_int32 * (act_scale * weight_scale)
            combined_scale = act_scale * self.weight_scale.float()
            output = output_int32.float() * combined_scale
        else:
            # Fallback: use FP matmul with dequantized weights
            output = self._fallback_forward(x_2d.float())

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.float()

        # Activation capture for MSE testing
        if _ACTIVATION_CAPTURE["enabled"]:
            should_capture = len(_ACTIVATION_CAPTURE["samples"]) < _ACTIVATION_CAPTURE["max_samples"]
            if should_capture:
                pattern = _ACTIVATION_CAPTURE["layer_name"]
                if pattern is None or (pattern and re.search(pattern, self.name)):
                    # Get dequantized FP16 weight
                    W_fp16 = self.qweight.float() * self.weight_scale[:, None].float()
                    _ACTIVATION_CAPTURE["samples"].append({
                        "layer_name": self.name,
                        "x": x_2d.detach().clone(),
                        "W_fp16": W_fp16.half().detach().clone(),
                        "weight_scale": self.weight_scale.detach().clone(),
                        "y_bitblas": output.detach().clone(),
                    })
                    logger.info(f"[W8A8] Captured activation {len(_ACTIVATION_CAPTURE['samples'])}/{_ACTIVATION_CAPTURE['max_samples']} from {self.name}")

        # Restore original shape and dtype
        output_shape = original_shape[:-1] + (self.out_features,)
        return output.view(output_shape).to(original_dtype)

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback forward using dequantized weights."""
        # Dequantize weights
        W = self.qweight.float() * self.weight_scale[:, None].float()
        return torch.nn.functional.linear(x, W)

    @property
    def weight(self) -> torch.Tensor:
        """Return dequantized weight for compatibility with code that accesses .weight"""
        if self.qweight is None:
            return None
        # Dequantize: W_fp16 = qweight * weight_scale
        W_fp16 = self.qweight.float() * self.weight_scale[:, None].float()
        return W_fp16.half()


def wrap_w8a8(
    model: nn.Module,
    layer_names: list,
    enable_tuning: bool = False,
    opt_M: list = None,
    act_scales: dict = None,
) -> int:
    """Replace Linear layers with BitBLASW8A8Linear.

    Args:
        model: The model to modify
        layer_names: List of layer names to replace
        enable_tuning: Whether to enable BitBLAS auto-tuning
        opt_M: Batch sizes to optimize for
        act_scales: Optional dict of {layer_name: activation_scale}

    Returns:
        Number of layers successfully replaced
    """
    replaced_count = 0

    for name in layer_names:
        # Navigate to parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        attr_name = parts[-1]
        linear = getattr(parent, attr_name)

        if not isinstance(linear, nn.Linear):
            logger.warning(f"[W8A8] {name}: Not a Linear layer, skipping")
            continue

        # Create W8A8 layer
        w8a8_linear = BitBLASW8A8Linear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            name=name,
            enable_tuning=enable_tuning,
            opt_M=opt_M,
        )

        # Load weights
        act_scale = act_scales.get(name) if act_scales else None
        w8a8_linear.load_from_linear(linear, act_scale=act_scale)

        # Move to same device
        w8a8_linear = w8a8_linear.to(linear.weight.device)

        # Replace
        setattr(parent, attr_name, w8a8_linear)
        replaced_count += 1

        if replaced_count <= 5 or replaced_count % 20 == 0:
            logger.info(f"[W8A8] Replaced {name}")

    return replaced_count


def select_targets_for_w8a8(
    model: nn.Module,
    include_regex: str,
    exclude_regex: str,
    scope_prefix: str = "",
) -> list:
    """Select Linear layers for W8A8 quantization based on regex patterns.

    Args:
        model: The model to analyze
        include_regex: Regex pattern for layers to include
        exclude_regex: Regex pattern for layers to exclude
        scope_prefix: Optional prefix to add to layer names

    Returns:
        List of layer names to quantize
    """
    layer_names = []

    include_pattern = re.compile(include_regex) if include_regex else None
    exclude_pattern = re.compile(exclude_regex) if exclude_regex else None

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        full_name = f"{scope_prefix}.{name}" if scope_prefix else name

        # Check include pattern
        if include_pattern and not include_pattern.search(full_name):
            continue

        # Check exclude pattern
        if exclude_pattern and exclude_pattern.search(full_name):
            continue

        layer_names.append(name)

    return layer_names


def get_w8a8_config_from_env() -> dict:
    """Get W8A8 configuration from environment variables."""
    opt_M_str = os.environ.get("OPENPI_W8A8_OPT_M", "1,16,32,64")
    opt_M = [int(x.strip()) for x in opt_M_str.split(",") if x.strip()]

    return {
        "enable_tuning": os.environ.get("OPENPI_W8A8_ENABLE_TUNING", "1") not in ("0", "false", "False"),
        "opt_M": opt_M,
        "include_regex": os.environ.get(
            "OPENPI_W8A8_INCLUDE",
            r"(.*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*|.*gemma_expert.*(gate_proj|up_proj|down_proj).*)"
        ),
        "exclude_regex": os.environ.get(
            "OPENPI_W8A8_EXCLUDE",
            r"(?:^|\.)(norm|ln|layernorm|emb|embed|vision_tower|multi_modal_projector|lm_head)(?:\.|$)"
        ),
        "debug": os.environ.get("OPENPI_W8A8_DEBUG", "0") not in ("0", "false", "False"),
    }


def enable_w8a8_if_configured(model: nn.Module) -> int:
    """Enable W8A8 quantization if OPENPI_W8A8_ENABLE=1.

    Args:
        model: The model to quantize

    Returns:
        Number of layers quantized, or 0 if not enabled
    """
    if os.environ.get("OPENPI_W8A8_ENABLE", "0") in ("0", "false", "False"):
        return 0

    config = get_w8a8_config_from_env()

    if config["debug"]:
        logger.info("[W8A8] Configuration from environment:")
        for k, v in config.items():
            logger.info(f"  {k}: {v}")

    return apply_w8a8_quantization(
        model,
        include_regex=config["include_regex"],
        exclude_regex=config["exclude_regex"],
        enable_tuning=config["enable_tuning"],
        opt_M=config["opt_M"],
    )


def apply_w8a8_quantization(
    model: nn.Module,
    include_regex: str = ".*",
    exclude_regex: str = r"(?:^|\.)(norm|ln|layernorm|emb|embed|vision_tower|multi_modal_projector|lm_head)(?:\.|$)",
    enable_tuning: bool = True,
    opt_M: list = None,
    act_scales: dict = None,
) -> int:
    """Apply W8A8 quantization to a model.

    Args:
        model: The model to quantize
        include_regex: Regex pattern for layers to include
        exclude_regex: Regex pattern for layers to exclude
        enable_tuning: Whether to enable BitBLAS auto-tuning
        opt_M: Batch sizes to optimize for
        act_scales: Optional dict of pre-calibrated activation scales

    Returns:
        Number of layers quantized
    """
    if opt_M is None:
        opt_M = [1, 16, 32, 64]

    logger.info("[W8A8] Starting W8A8 quantization")
    logger.info(f"[W8A8] Include pattern: {include_regex}")
    logger.info(f"[W8A8] Exclude pattern: {exclude_regex}")
    logger.info(f"[W8A8] Tuning enabled: {enable_tuning}")

    # Try to load previously tuned operators from database
    cached_count = load_tuning_cache()
    if cached_count > 0:
        logger.info(f"[W8A8] Will use {cached_count} cached tuned operators")

    # Select target layers
    layer_names = select_targets_for_w8a8(model, include_regex, exclude_regex)

    logger.info(f"[W8A8] Found {len(layer_names)} layers to quantize")
    if len(layer_names) > 0:
        for name in layer_names[:10]:
            logger.info(f"[W8A8]   - {name}")
        if len(layer_names) > 10:
            logger.info(f"[W8A8]   ... and {len(layer_names) - 10} more")

    # Replace layers
    replaced_count = wrap_w8a8(
        model,
        layer_names,
        enable_tuning=enable_tuning,
        opt_M=opt_M,
        act_scales=act_scales,
    )

    logger.info(f"[W8A8] Successfully replaced {replaced_count}/{len(layer_names)} layers")

    # Save tuned operators to database for future use
    if enable_tuning:
        save_tuning_cache()

    return replaced_count
