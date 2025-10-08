import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import importlib
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch import nn

from .duquant_preprocess import (
    PackResult,
    PercentileCalibrator,
    apply_input_transform,
    apply_output_restore,
    apply_bias_row_rot,
    fake_quantize_sym,
    load_pack,
    pack_weight,
    qmax,
    save_pack,
    transform_weight_for_forward,
)
from .quant_backends import QLinearW4A8BitBLAS


@dataclass
class DuQuantConfig:
    weight_bits: int = int(os.environ.get("OPENPI_DUQUANT_WBITS_DEFAULT", 4))
    act_bits: int = int(os.environ.get("OPENPI_DUQUANT_ABITS", 8))
    block_size: int = int(os.environ.get("OPENPI_DUQUANT_BLOCK", 16))
    lambda_smooth: float = float(os.environ.get("OPENPI_DUQUANT_LS", 0.15))
    enable_permute: bool = os.environ.get("OPENPI_DUQUANT_PERMUTE", "1") not in ("0", "false", "False")
    act_percentile: float = float(os.environ.get("OPENPI_DUQUANT_ACT_PCT", 99.9))
    calib_batches: int = int(os.environ.get("OPENPI_DUQUANT_CALIB_STEPS", 32))
    pack_dir: Optional[str] = os.environ.get("OPENPI_DUQUANT_PACKDIR", None)
    row_rot_mode: str = os.environ.get("OPENPI_DUQUANT_ROW_ROT", "restore")  # values: '0', 'restore', 'propagate'
    block_out_size: int = int(os.environ.get("OPENPI_DUQUANT_BLOCK_OUT", os.environ.get("OPENPI_DUQUANT_BLOCK", 16)))


def _load_bitblas_fallback():
    try:
        module = importlib.import_module("openpi.models_pytorch.quant_backends.bitblas_fallback")
    except Exception:
        return None
    sys.modules["bitblas"] = module  # type: ignore[assignment]
    print("[DUQUANT][INFO] Using fallback BitBLAS stub implementation.")
    return module


def _bitblas_is_available() -> bool:
    try:
        bitblas = importlib.import_module("bitblas")  # type: ignore
    except Exception:
        module = _load_bitblas_fallback()
        return hasattr(module, "linear_w4a8") if module is not None else False

    if hasattr(bitblas, "linear_w4a8"):
        return True
    ops = getattr(bitblas, "ops", None)
    if ops is not None and hasattr(ops, "linear_w4a8"):
        return True

    module = _load_bitblas_fallback()
    return hasattr(module, "linear_w4a8") if module is not None else False


def _load_or_create_pack(mod: nn.Linear, name: str, cfg: DuQuantConfig) -> PackResult:
    pack = load_pack(name, cfg.pack_dir)
    if pack is None:
        pack = pack_weight(
            mod.weight.detach(),
            block_size=cfg.block_size,
            block_out_size=cfg.block_out_size,
            enable_permute=cfg.enable_permute,
            lambda_smooth=cfg.lambda_smooth,
        )
        save_pack(name, pack, cfg.pack_dir)
    return pack


def _parse_per_layer_wbits(env_val: Optional[str]) -> Dict[str, int]:
    if not env_val:
        return {}
    result: Dict[str, int] = {}
    parts = [p.strip() for p in env_val.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        try:
            result[k.strip()] = int(v.strip())
        except ValueError:
            pass
    return result


class DuQuantLinear(nn.Module):
    def __init__(self, base: nn.Linear, name: str, cfg: DuQuantConfig, weight_bits: Optional[int] = None) -> None:
        super().__init__()
        self.name = name
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None
        self.register_buffer("_weight", base.weight.detach().clone())

        # Config
        self.cfg = cfg
        self.weight_bits = cfg.weight_bits if weight_bits is None else int(weight_bits)

        # Load or compute packing
        pack = load_pack(self.name, cfg.pack_dir)
        if pack is None:
            pack = pack_weight(
                self._weight,
                block_size=cfg.block_size,
                block_out_size=cfg.block_out_size,
                enable_permute=cfg.enable_permute,
                lambda_smooth=cfg.lambda_smooth,
            )
            save_pack(self.name, pack, cfg.pack_dir)
        self.pack: PackResult = pack

        # ========================================
        # OPTIMIZATION: Pre-cache rotation matrices as torch tensors
        # This avoids torch.from_numpy() and .to(device) in every forward pass
        # ========================================
        # Cache permutation tensor
        if pack.perm is not None:
            self.register_buffer("_perm_cache", torch.from_numpy(pack.perm).long())
        else:
            self._perm_cache = None

        # Cache input rotation matrices (R_in_blocks)
        # Store block indices, not tensor references (tensors auto-follow device via register_buffer)
        self._R_in_block_indices: List[int] = []
        if pack.R_in_blocks:
            for b, R in pack.R_in_blocks.items():
                # Register as buffer so it moves with model to device
                # Use weight dtype to match model precision (float32/bfloat16/etc)
                buffer_name = f"_R_in_{b}"
                self.register_buffer(buffer_name, torch.from_numpy(R).to(dtype=self._weight.dtype))
                self._R_in_block_indices.append(b)

        # Cache output rotation matrices (R_out_blocks)
        self._R_out_block_indices: List[int] = []
        if pack.R_out_blocks:
            for b, R in pack.R_out_blocks.items():
                buffer_name = f"_R_out_{b}"
                self.register_buffer(buffer_name, torch.from_numpy(R).to(dtype=self._weight.dtype))
                self._R_out_block_indices.append(b)

        # Store metadata needed for fast paths
        self._block_size = int(pack.meta.get("block_size", 16))
        self._block_out_size = int(pack.meta.get("block_out_size", self._block_size))
        # ========================================

        # Calibrator for activation
        self.calibrator = PercentileCalibrator(
            percentile=cfg.act_percentile, max_batches=cfg.calib_batches
        ) if self.cfg.act_bits > 0 else None
        self._act_scale: Optional[torch.Tensor] = None

        # Cache transformed weight per device/dtype
        self._cached_weight_key: Optional[Tuple[str, torch.dtype]] = None
        self.register_buffer("_W_t", torch.zeros_like(self._weight))
        self.register_buffer("_w_scales", torch.ones(self.out_features, dtype=self._weight.dtype))
        # OPTIMIZATION: Pre-cache quantized weights
        self.register_buffer("_W_t_quantized", torch.zeros_like(self._weight))
        self._weight_quantized_cached = False

        self._bias_rot: Optional[torch.Tensor] = None
        self._debug_enabled = os.environ.get("OPENPI_DUQUANT_DEBUG", "0") not in ("0", "false", "False")
        self._debug_forward_logged = False

    def _get_R_in_cache(self) -> Dict[int, torch.Tensor]:
        """Get R_in rotation matrices on the correct device."""
        # Cache the dict to avoid recreating it every forward pass
        # This helps torch.compile recognize the same computation graph
        if not hasattr(self, '_R_in_cache_dict'):
            self._R_in_cache_dict = {}
        # Update references (buffers may have moved to different device)
        for b in self._R_in_block_indices:
            self._R_in_cache_dict[b] = getattr(self, f"_R_in_{b}")
        return self._R_in_cache_dict

    def _get_R_out_cache(self) -> Dict[int, torch.Tensor]:
        """Get R_out rotation matrices on the correct device."""
        if not hasattr(self, '_R_out_cache_dict'):
            self._R_out_cache_dict = {}
        for b in self._R_out_block_indices:
            self._R_out_cache_dict[b] = getattr(self, f"_R_out_{b}")
        return self._R_out_cache_dict

    @property
    def weight(self) -> torch.Tensor:
        """Expose packed weight buffer for compatibility with existing code paths."""
        return self._weight

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:  # pragma: no cover - defensive
        with torch.no_grad():
            self._weight.copy_(value)

    def _maybe_update_weight_cache(self) -> None:
        apply_row = (self.cfg.row_rot_mode != "0")
        key = (str(self._weight.device), self._weight.dtype, int(self.weight_bits), int(apply_row))
        if self._cached_weight_key == key:
            return

        # Import here to use optimized version
        from .duquant_preprocess import transform_weight_for_forward_optimized

        W_t, scales = transform_weight_for_forward_optimized(
            self._weight,
            self.pack,
            weight_bits=self.weight_bits,
            apply_row_rot=apply_row,
            perm_cache=self._perm_cache,
            R_in_cache=self._get_R_in_cache(),  # Get tensors on correct device
            R_out_cache=self._get_R_out_cache(),  # Get tensors on correct device
            block_size=self._block_size,
            block_out_size=self._block_out_size,
        )
        # in-place copy to preserve buffers and state_dict tracking
        self._W_t.copy_(W_t)
        self._w_scales.copy_(scales)

        # OPTIMIZATION: Pre-quantize weights if weight_bits > 0
        if self.weight_bits > 0:
            with torch.no_grad():
                self._W_t_quantized.copy_(
                    fake_quantize_sym(W_t, scales[:, None], self.weight_bits, label="weight_prequant")
                )
            self._weight_quantized_cached = True
        else:
            self._weight_quantized_cached = False

        self._cached_weight_key = key
        if self.bias is not None:
            if self.cfg.row_rot_mode == "propagate" and self.pack.R_out_blocks is not None:
                with torch.no_grad():
                    # Use optimized version with cached tensors
                    from .duquant_preprocess import apply_bias_row_rot_optimized
                    self._bias_rot = apply_bias_row_rot_optimized(
                        self.bias.detach(), self.pack, self._get_R_out_cache(), self._block_out_size
                    )
            else:
                self._bias_rot = None
        if self._debug_enabled:
            # Avoid .item() which causes graph breaks in torch.compile
            import logging
            logging.info(
                f"[DUQUANT][CACHE] {self.name} device={self._weight.device} dtype={self._weight.dtype} "
                f"Wbits={self.weight_bits} Abits={self.cfg.act_bits} block_in={self.cfg.block_size} "
                f"permute={self.pack.perm is not None} row_rot={self.cfg.row_rot_mode}"
            )
            if self._weight_quantized_cached:
                logging.info(f"[DUQUANT][CACHE] {self.name} pre-quantized weights cached")

    def _get_act_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.act_bits <= 0:
            return torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
        if self._act_scale is not None:
            return self._act_scale
        if self.calibrator is not None and not self.calibrator.is_full():
            self.calibrator.observe(x)
            if self.calibrator.is_full():
                p_vec = self.calibrator.finalize()
                max_q = qmax(self.cfg.act_bits)
                scale = torch.clamp(p_vec / max_q, min=1e-6)
                self._act_scale = scale.to(dtype=x.dtype, device=x.device)
        # Fallback single-batch observation
        if self._act_scale is None:
            with torch.no_grad():
                x_abs = torch.abs(x.detach().to(torch.float32))
                C = x_abs.shape[-1]
                x2d = x_abs.reshape(-1, C)
                p_vec = torch.quantile(x2d, self.cfg.act_percentile / 100.0, dim=0)
                max_q = qmax(self.cfg.act_bits)
                scale = torch.clamp(p_vec / max_q, min=1e-6)
                self._act_scale = scale.to(dtype=x.dtype, device=x.device)
        return self._act_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Import optimized functions
        from .duquant_preprocess import apply_input_transform_optimized, apply_output_restore_optimized

        # Apply inverse transforms to inputs using optimized version
        x_t = apply_input_transform_optimized(
            x, self.pack, self._perm_cache, self._get_R_in_cache(), self._block_size
        )

        # Fake-quantize activations if enabled
        if self.cfg.act_bits > 0:
            s_a = self._get_act_scale(x_t)
            x_t = fake_quantize_sym(x_t, s_a, self.cfg.act_bits, label="activation_forward")

        # Transform and fake-quantize weights (only once, cached)
        self._maybe_update_weight_cache()

        # OPTIMIZATION: Use pre-quantized weights instead of quantizing every forward pass
        if self._weight_quantized_cached:
            # Use pre-quantized weights (major speedup)
            y_lin = torch.nn.functional.linear(x_t, self._W_t_quantized, None)
        elif self.weight_bits > 0:
            # Fallback: quantize on-the-fly (slower, should not happen after warmup)
            y_lin = torch.nn.functional.linear(
                x_t,
                fake_quantize_sym(
                    self._W_t,
                    self._w_scales[:, None],
                    self.weight_bits,
                    label="weight_fallback",
                ),
                None
            )
        else:
            y_lin = torch.nn.functional.linear(x_t, self._W_t, None)

        # Apply row restore if requested using optimized version
        if self.cfg.row_rot_mode == "restore" and self.pack.R_out_blocks is not None:
            y_lin = apply_output_restore_optimized(
                y_lin, self.pack, self._get_R_out_cache(), self._block_out_size
            )
            # add bias after restore to preserve exact equivalence
            if self.bias is not None:
                y_lin = y_lin + self.bias
        else:
            # propagate or disabled: keep bias in current basis
            if self.bias is not None:
                bias_to_add = (
                    self._bias_rot
                    if self.cfg.row_rot_mode == "propagate" and self._bias_rot is not None
                    else self.bias
                )
                y_lin = y_lin + bias_to_add
        if self._debug_enabled and not self._debug_forward_logged:
            import logging
            logging.info(
                f"[DUQUANT][FORWARD] {self.name} input={tuple(x.shape)} output={tuple(y_lin.shape)} "
                f"weight_bits={self.weight_bits} act_bits={self.cfg.act_bits}"
            )
            self._debug_forward_logged = True
        return y_lin


def _get_parent_module_and_attr(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def select_targets(
    model: nn.Module,
    *,
    include_regex: str = r".*(q_proj|k_proj|v_proj|out_proj|fc1|fc2|up_proj|down_proj).*",
    exclude_regex: str = r"(?:^|\.)(norm|ln|layernorm|emb)(?:\.|$)",
    scope_prefix: Optional[str] = None,
    whitelist: Optional[Iterable[str]] = None,
    blacklist: Optional[Iterable[str]] = None,
) -> List[Tuple[str, nn.Linear]]:
    inc = re.compile(include_regex)
    exc = re.compile(exclude_regex)
    wl = set(whitelist or [])
    bl = set(blacklist or [])
    results: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if scope_prefix is not None and not name.startswith(scope_prefix):
            continue
        if name in bl:
            continue
        if wl and name not in wl:
            continue
        if not wl and (not inc.search(name) or exc.search(name)):
            continue
        results.append((name, mod))
    return results


def wrap_duquant(
    model: nn.Module,
    layer_names: Iterable[str],
    cfg: DuQuantConfig,
    per_layer_wbits: Optional[Dict[str, int]] = None,
    dry_run: bool = False,
    backend: str = "fake",
    bitblas_available: bool = False,
) -> None:
    per_layer_wbits = per_layer_wbits or {}
    replaced = 0
    replaced_by_backend: Dict[str, int] = {"fake": 0, "bitblas": 0}
    listed = 0
    for name in layer_names:
        # Skip action head by default unless explicitly requested via OPENPI_DUQUANT_INCLUDE
        if os.environ.get("OPENPI_DUQUANT_INCLUDE_ACTION_HEAD", "0") in ("0", "false", "False"):
            if name.endswith("action_out_proj") or ".action_out_proj" in name:
                continue
        parent, attr = _get_parent_module_and_attr(model, name)
        mod = getattr(parent, attr)
        if not isinstance(mod, nn.Linear):
            continue
        wbits = per_layer_wbits.get(name, cfg.weight_bits)

        desired_backend = backend
        if backend == "auto":
            desired_backend = "bitblas" if bitblas_available else "fake"

        if desired_backend == "bitblas" and wbits != 4:
            if backend == "auto":
                print(
                    f"[DUQUANT][AUTO] {name}: requested W{wbits}, falling back to fake backend (BitBLAS supports W4 only)"
                )
                desired_backend = "fake"
            else:
                raise ValueError(f"[DUQUANT] BitBLAS backend currently supports only 4-bit weights (layer {name})")

        if dry_run:
            msg = (
                f"[DUQUANT][DRYRUN] {name}: Linear({mod.in_features}->{mod.out_features}) "
                f"W{wbits} A{cfg.act_bits} perm={cfg.enable_permute} "
                f"block_in={cfg.block_size} block_out={cfg.block_out_size} row_rot={cfg.row_rot_mode} "
                f"backend={desired_backend}"
            )
            print(msg)
            listed += 1
            continue

        backend_used = desired_backend
        dq_module: nn.Module
        if desired_backend == "bitblas":
            pack = _load_or_create_pack(mod, name, cfg)
            try:
                dq_module = QLinearW4A8BitBLAS(
                    mod, name=name, cfg=cfg, pack=pack, weight_bits=wbits
                )
            except Exception as exc:
                if backend == "auto":
                    print(
                        f"[DUQUANT][AUTO] {name}: BitBLAS backend unavailable ({exc!r}), falling back to fake."
                    )
                    backend_used = "fake"
                else:
                    raise
            else:
                setattr(parent, attr, dq_module)
                print(
                    f"[DUQUANT][REPLACED] {name}: Linear({mod.in_features}->{mod.out_features}) -> QLinearW4A8BitBLAS "
                    f"W{wbits} A{cfg.act_bits} block_in={cfg.block_size} block_out={cfg.block_out_size}"
                )
                replaced += 1
                replaced_by_backend["bitblas"] += 1
                continue

        if backend_used != "bitblas":
            dq_module = DuQuantLinear(mod, name=name, cfg=cfg, weight_bits=wbits)
            setattr(parent, attr, dq_module)
            print(
                f"[DUQUANT][REPLACED] {name}: Linear({mod.in_features}->{mod.out_features}) -> DuQuantLinear "
                f"W{wbits} A{cfg.act_bits} perm={cfg.enable_permute} block_in={cfg.block_size} block_out={cfg.block_out_size} row_rot={cfg.row_rot_mode}"
            )
            replaced += 1
            replaced_by_backend["fake"] += 1

    if dry_run:
        print(f"[DUQUANT] Dry-run total layers listed: {listed}")
    else:
        bitblas_count = replaced_by_backend.get("bitblas", 0)
        fake_count = replaced_by_backend.get("fake", 0)
        summary = (
            f"[DUQUANT] Total layers replaced: {replaced} "
            f"(bitblas={bitblas_count}, fake={fake_count})"
        )
        print(summary)


def enable_duquant_if_configured(model: nn.Module) -> None:
    """Entry point to enable DuQuant based on environment variables.

    Activation conditions:
    - If OPENPI_DUQUANT_DRYRUN is set => dry-run listing only
    - Or if any OPENPI_DUQUANT_* variable (other than PACKDIR) is set => perform replacement
    - Otherwise do nothing
    """
    env = os.environ
    keys = [k for k in env.keys() if k.startswith("OPENPI_DUQUANT_")]
    activate = any(k not in ("OPENPI_DUQUANT_PACKDIR",) for k in keys)
    if not activate:
        return

    # Scope defaults to the DiT backbone path under policy
    scope = env.get("OPENPI_DUQUANT_SCOPE", "policy.dit.")
    whitelist = env.get("OPENPI_DUQUANT_LAYERS")
    whitelist_list = [x.strip() for x in whitelist.split(",") if x.strip()] if whitelist else None
    inc = env.get("OPENPI_DUQUANT_INCLUDE", r".*(q_proj|k_proj|v_proj|o_proj|out_proj|fc1|fc2|gate_proj|up_proj|down_proj).*")
    exc = env.get("OPENPI_DUQUANT_EXCLUDE", r"(?:^|\.)(norm|ln|layernorm|emb)(?:\.|$)")
    per_layer_wbits = _parse_per_layer_wbits(env.get("OPENPI_DUQUANT_WBITS"))
    dry_run = env.get("OPENPI_DUQUANT_DRYRUN", "0") not in ("0", "false", "False")

    backend = env.get("OPENPI_DUQUANT_BACKEND", "fake").lower()
    valid_backends = {"off", "fake", "bitblas", "auto"}
    if backend not in valid_backends:
        print(f"[DUQUANT] Unknown backend '{backend}', defaulting to 'fake'")
        backend = "fake"
    if backend == "off":
        print("[DUQUANT] Backend set to 'off'; skipping DuQuant replacement.")
        return

    bitblas_available = _bitblas_is_available()
    if backend == "bitblas" and not bitblas_available:
        raise RuntimeError(
            "[DUQUANT] OPENPI_DUQUANT_BACKEND=bitblas but BitBLAS kernels are unavailable. "
            "Install BitBLAS or set backend to 'fake' or 'auto'."
        )

    cfg = DuQuantConfig()

    targets = select_targets(
        model,
        include_regex=inc,
        exclude_regex=exc,
        scope_prefix=scope,
        whitelist=whitelist_list,
        blacklist=None,
    )
    layer_names = [n for n, _ in targets]
    print(f"[DUQUANT] SCOPE filter: '{scope}'")
    print(f"[DUQUANT] Matched Linear layers: {len(layer_names)}")
    if len(layer_names) == 0 and scope:
        # Debug: print some layer names to help diagnose
        all_linears = [(n, m) for n, m in model.named_modules() if isinstance(m, __import__('torch').nn.Linear)]
        print(f"[DUQUANT] DEBUG: Total Linear layers in model: {len(all_linears)}")
        print(f"[DUQUANT] DEBUG: First 10 Linear layer names:")
        for name, _ in all_linears[:10]:
            print(f"[DUQUANT] DEBUG:   {name}")
        if scope:
            matching_prefix = [n for n, _ in all_linears if n.startswith(scope.rstrip('.'))]
            print(f"[DUQUANT] DEBUG: Layers matching prefix '{scope.rstrip('.')}': {len(matching_prefix)}")
            if matching_prefix:
                for name in matching_prefix[:5]:
                    print(f"[DUQUANT] DEBUG:   {name}")
    if dry_run:
        wrap_duquant(
            model,
            layer_names,
            cfg,
            per_layer_wbits,
            dry_run=True,
            backend=backend,
            bitblas_available=bitblas_available,
        )
        return
    wrap_duquant(
        model,
        layer_names,
        cfg,
        per_layer_wbits,
        dry_run=False,
        backend=backend,
        bitblas_available=bitblas_available,
    )
