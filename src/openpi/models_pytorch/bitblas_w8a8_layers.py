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


# =============================================================================
# HuggingFace Save/Load Functions
# =============================================================================

def save_w8a8_to_hf(
    policy,
    save_path: str,
    checkpoint_dir: str,
    repo_id: Optional[str] = None,
    push_to_hub: bool = False,
):
    """Save complete W8A8 quantized model to HuggingFace format.

    Saves the complete model including:
    - INT8 weights (qweight) and scales (weight_scale) for W8A8 layers
    - FP16 weights for non-quantized layers (vision encoder, embeddings, etc.)
    - assets directory (norm_stats for inference)

    This allows users to download a single model and run inference directly.

    Args:
        policy: The policy object containing the quantized model
        save_path: Local path to save the model
        checkpoint_dir: Path to original checkpoint (for copying assets)
        repo_id: HuggingFace repo ID (e.g., "username/pi05-libero-w8a8")
        push_to_hub: Whether to upload to HuggingFace Hub

    Example:
        >>> policy = create_trained_policy(config, checkpoint_dir)
        >>> save_w8a8_to_hf(
        ...     policy,
        ...     "./pi05-libero-w8a8",
        ...     checkpoint_dir="~/VLM_REPO/openpi/ckpts/pi05_libero_torch",
        ...     repo_id="username/pi05-libero-w8a8",
        ...     push_to_hub=True
        ... )
    """
    import json
    import shutil
    from safetensors.torch import save_file

    os.makedirs(save_path, exist_ok=True)
    checkpoint_dir = os.path.expanduser(checkpoint_dir)

    # Get the model from policy
    model = policy._model if hasattr(policy, "_model") else policy

    # Collect COMPLETE state dict - all parameters
    state_dict = {}
    w8a8_layer_names = []

    # First, collect all non-module parameters (embeddings, norms, etc.)
    for name, param in model.named_parameters():
        # Skip if this belongs to a W8A8 layer (will be handled separately)
        is_w8a8_param = False
        for mod_name, mod in model.named_modules():
            if isinstance(mod, BitBLASW8A8Linear) and name.startswith(mod_name + "."):
                is_w8a8_param = True
                break
        if not is_w8a8_param:
            state_dict[name] = param.data

    # Collect buffers (like position embeddings, etc.)
    for name, buffer in model.named_buffers():
        # Skip W8A8 layer buffers
        is_w8a8_buffer = False
        for mod_name, mod in model.named_modules():
            if isinstance(mod, BitBLASW8A8Linear) and name.startswith(mod_name + "."):
                is_w8a8_buffer = True
                break
        if not is_w8a8_buffer:
            state_dict[name] = buffer

    # Now handle W8A8 layers specially
    for name, module in model.named_modules():
        if isinstance(module, BitBLASW8A8Linear):
            if module.qweight is not None:
                # Get dequantized weight and requantize to get clean INT8
                W_fp16 = module.weight  # This returns dequantized weight
                weight_absmax = W_fp16.float().abs().max(dim=1)[0]
                weight_scale = (weight_absmax / 127.0).clamp(min=1e-8)
                W_int8 = (W_fp16.float() / weight_scale[:, None]).round().clamp(-127, 127).to(torch.int8)

                state_dict[f"{name}.qweight"] = W_int8
                state_dict[f"{name}.weight_scale"] = weight_scale.half()

                if module.bias is not None:
                    state_dict[f"{name}.bias"] = module.bias

                w8a8_layer_names.append(name)

    # Save state dict as safetensors
    save_file(state_dict, os.path.join(save_path, "model.safetensors"))
    logger.info(f"[W8A8] Saved {len(state_dict)} tensors ({len(w8a8_layer_names)} W8A8 layers) to {save_path}/model.safetensors")

    # Save W8A8 config
    config = {
        "quantization": "w8a8_bitblas",
        "w8a8_layers": w8a8_layer_names,
        "model_type": type(model).__name__,
        "complete_model": True,  # Flag indicating this is a complete model
    }

    with open(os.path.join(save_path, "w8a8_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"[W8A8] Saved config to {save_path}/w8a8_config.json")

    # Copy assets directory from original checkpoint
    src_assets = os.path.join(checkpoint_dir, "assets")
    dst_assets = os.path.join(save_path, "assets")
    if os.path.exists(src_assets):
        if os.path.exists(dst_assets):
            shutil.rmtree(dst_assets)
        shutil.copytree(src_assets, dst_assets)
        logger.info(f"[W8A8] Copied assets from {src_assets} to {dst_assets}")
    else:
        logger.warning(f"[W8A8] Assets directory not found: {src_assets}")

    # Push to HuggingFace Hub if requested
    if push_to_hub and repo_id:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.upload_folder(
                folder_path=save_path,
                repo_id=repo_id,
                repo_type="model",
            )
            logger.info(f"[W8A8] Uploaded to HuggingFace: {repo_id}")
        except Exception as e:
            logger.error(f"[W8A8] Failed to upload to HuggingFace: {e}")
            raise

    return save_path


def load_w8a8_policy(
    repo_id_or_path: str,
    policy_config_name: str = "pi05_libero",
    enable_tuning: bool = False,
    opt_M: list = None,
    device: str = "cuda",
):
    """Load complete W8A8 quantized model from HuggingFace or local path.

    This function loads a complete W8A8 model that includes:
    - INT8 quantized weights for LLM/DiT layers
    - FP16 weights for vision encoder and other non-quantized layers
    - assets directory with norm_stats

    Users can download the model and run inference directly without needing
    the original FP16 checkpoint.

    Args:
        repo_id_or_path: HuggingFace repo ID (e.g., "fatdove/pi05-libero-w8a8")
                        or local path to saved model
        policy_config_name: Name of the policy config (default: "pi05_libero")
        enable_tuning: Whether to enable BitBLAS tuning
        opt_M: Batch sizes to optimize for
        device: Device to load model on

    Returns:
        Policy object ready for inference

    Example:
        >>> policy = load_w8a8_policy("fatdove/pi05-libero-w8a8")
        >>> result = policy.infer(observation)
    """
    import json
    import safetensors.torch
    from safetensors.torch import load_file

    if opt_M is None:
        opt_M = [1, 16, 32, 64]

    # Step 1: Download from HuggingFace or use local path
    if os.path.exists(repo_id_or_path):
        w8a8_model_path = repo_id_or_path
        logger.info(f"[W8A8] Loading from local path: {w8a8_model_path}")
    else:
        try:
            from huggingface_hub import snapshot_download

            w8a8_model_path = snapshot_download(repo_id=repo_id_or_path)
            logger.info(f"[W8A8] Downloaded from HuggingFace: {repo_id_or_path}")
        except Exception as e:
            raise ValueError(f"Failed to download from HuggingFace: {e}") from e

    # Step 2: Load W8A8 config
    config_path = os.path.join(w8a8_model_path, "w8a8_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            w8a8_config = json.load(f)
        w8a8_layer_names = w8a8_config.get("w8a8_layers", [])
        is_complete_model = w8a8_config.get("complete_model", False)
    else:
        w8a8_layer_names = []
        is_complete_model = False
        logger.warning("[W8A8] No w8a8_config.json found")

    if not is_complete_model:
        raise ValueError(
            "This model is not a complete W8A8 model. "
            "Please use a model saved with save_w8a8_to_hf() which includes all weights."
        )

    # Step 3: Create model and load weights from W8A8 checkpoint
    from openpi.training import config as _config
    from openpi.models_pytorch import pi0_pytorch

    train_config = _config.get_config(policy_config_name)

    # Create empty model structure
    model = pi0_pytorch.PI0Pytorch(config=train_config.model)

    # Load the complete state dict
    weight_path = os.path.join(w8a8_model_path, "model.safetensors")
    state_dict = load_file(weight_path)
    logger.info(f"[W8A8] Loaded state dict with {len(state_dict)} tensors")

    # Auto-detect W8A8 layers from state dict if not in config
    if not w8a8_layer_names:
        w8a8_layer_names = [
            key.rsplit(".qweight", 1)[0]
            for key in state_dict.keys()
            if key.endswith(".qweight")
        ]

    # Step 4: Load non-W8A8 weights first
    non_w8a8_state_dict = {}
    for key, value in state_dict.items():
        # Skip W8A8 layer weights (qweight, weight_scale)
        is_w8a8_key = any(
            key.startswith(name + ".qweight") or key.startswith(name + ".weight_scale")
            for name in w8a8_layer_names
        )
        if not is_w8a8_key:
            non_w8a8_state_dict[key] = value

    # Load non-W8A8 weights into model
    missing, unexpected = model.load_state_dict(non_w8a8_state_dict, strict=False)
    logger.info(f"[W8A8] Loaded {len(non_w8a8_state_dict)} non-W8A8 tensors")
    if missing:
        # Filter out expected missing keys (W8A8 layer weights)
        truly_missing = [k for k in missing if not any(k.startswith(name) for name in w8a8_layer_names)]
        if truly_missing:
            logger.warning(f"[W8A8] Missing {len(truly_missing)} keys: {truly_missing[:5]}...")

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model = model.to(device)

    # Step 5: Replace Linear layers with BitBLASW8A8Linear
    for layer_name in w8a8_layer_names:
        # Get the Linear layer
        parts = layer_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]
        linear = getattr(parent, attr_name)

        if not isinstance(linear, nn.Linear):
            logger.warning(f"[W8A8] {layer_name}: Expected Linear, got {type(linear)}")
            continue

        # Create BitBLASW8A8Linear
        w8a8_layer = BitBLASW8A8Linear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            name=layer_name,
            enable_tuning=enable_tuning,
            opt_M=opt_M,
        )

        # Load INT8 weights
        qweight_key = f"{layer_name}.qweight"
        scale_key = f"{layer_name}.weight_scale"

        if qweight_key in state_dict and scale_key in state_dict:
            qweight = state_dict[qweight_key].to(device)
            weight_scale = state_dict[scale_key].to(device)

            # Transform weight for BitBLAS if available
            if w8a8_layer.bitblas_matmul is not None:
                try:
                    transformed = w8a8_layer.bitblas_matmul.transform_weight(qweight)
                    if isinstance(transformed, list):
                        transformed = transformed[0]
                    w8a8_layer.qweight = transformed
                except Exception as e:
                    logger.warning(f"[W8A8] {layer_name}: transform_weight failed: {e}")
                    w8a8_layer.qweight = qweight
            else:
                w8a8_layer.qweight = qweight

            w8a8_layer.weight_scale = weight_scale

            # Load bias if present
            bias_key = f"{layer_name}.bias"
            if bias_key in state_dict:
                w8a8_layer.bias = state_dict[bias_key].to(device)

            # Replace in model
            setattr(parent, attr_name, w8a8_layer)

    logger.info(f"[W8A8] Replaced {len(w8a8_layer_names)} Linear layers with BitBLASW8A8Linear")

    # Step 6: Create policy object
    from openpi.training import checkpoints as _checkpoints
    import openpi.policies.policy as _policy
    import openpi.transforms as transforms

    # Load norm stats from W8A8 model's assets directory
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = None

    # Try loading from W8A8 model's assets directory
    assets_path = os.path.join(w8a8_model_path, "assets")
    if os.path.exists(assets_path) and data_config.asset_id:
        try:
            norm_stats = _checkpoints.load_norm_stats(assets_path, data_config.asset_id)
            logger.info(f"[W8A8] Loaded norm_stats from {assets_path}")
        except Exception as e:
            logger.warning(f"[W8A8] Could not load norm_stats from {assets_path}: {e}")

    if norm_stats is None:
        logger.warning("[W8A8] norm_stats not found, using empty dict")
        norm_stats = {}

    policy = _policy.Policy(
        model,
        transforms=[
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs=None,
        metadata=train_config.policy_metadata,
        is_pytorch=True,
        pytorch_device=device,
    )

    # Mark as W8A8 loaded
    policy._is_w8a8 = True
    policy._w8a8_layer_count = len(w8a8_layer_names)

    logger.info(f"[W8A8] Policy created with {len(w8a8_layer_names)} W8A8 layers")

    return policy
