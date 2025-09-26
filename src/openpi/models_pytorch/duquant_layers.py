import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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

        # Calibrator for activation
        self.calibrator = PercentileCalibrator(
            percentile=cfg.act_percentile, max_batches=cfg.calib_batches
        ) if self.cfg.act_bits > 0 else None
        self._act_scale: Optional[torch.Tensor] = None

        # Cache transformed weight per device/dtype
        self._cached_weight_key: Optional[Tuple[str, torch.dtype]] = None
        self.register_buffer("_W_t", torch.zeros_like(self._weight))
        self.register_buffer("_w_scales", torch.ones(self.out_features, dtype=self._weight.dtype))
        self._bias_rot: Optional[torch.Tensor] = None
        self._debug_enabled = os.environ.get("OPENPI_DUQUANT_DEBUG", "0") not in ("0", "false", "False")
        self._debug_forward_logged = False

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
        W_t, scales = transform_weight_for_forward(
            self._weight,
            self.pack,
            weight_bits=self.weight_bits,
            apply_row_rot=apply_row,
        )
        # in-place copy to preserve buffers and state_dict tracking
        self._W_t.copy_(W_t)
        self._w_scales.copy_(scales)
        self._cached_weight_key = key
        if self.bias is not None:
            if self.cfg.row_rot_mode == "propagate" and self.pack.R_out_blocks is not None:
                with torch.no_grad():
                    self._bias_rot = apply_bias_row_rot(self.bias.detach(), self.pack)
            else:
                self._bias_rot = None
        if self._debug_enabled:
            print(
                f"[DUQUANT][CACHE] {self.name} device={self._weight.device} dtype={self._weight.dtype} "
                f"Wbits={self.weight_bits} Abits={self.cfg.act_bits} block_in={self.cfg.block_size} "
                f"permute={self.pack.perm is not None} row_rot={self.cfg.row_rot_mode}"
            )
            print(
                f"[DUQUANT][CACHE] {self.name} weight_scales shape={tuple(self._w_scales.shape)} "
                f"min={self._w_scales.min().item():.4e} max={self._w_scales.max().item():.4e}"
            )
            if self._bias_rot is not None:
                print(
                    f"[DUQUANT][CACHE] {self.name} bias rotated shape={tuple(self._bias_rot.shape)}"
                )

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
        # Apply inverse transforms to inputs
        x_t = apply_input_transform(x, self.pack)
        # Fake-quantize activations if enabled
        if self.cfg.act_bits > 0:
            s_a = self._get_act_scale(x_t)
            x_t = fake_quantize_sym(x_t, s_a, self.cfg.act_bits)
        # Transform and fake-quantize weights
        self._maybe_update_weight_cache()
        W_t = self._W_t
        if self.weight_bits > 0:
            # Per-output channel fake quantization
            y_lin = torch.nn.functional.linear(
                x_t, fake_quantize_sym(W_t, self._w_scales[:, None], self.weight_bits), None
            )
        else:
            y_lin = torch.nn.functional.linear(x_t, W_t, None)

        # Apply row restore if requested
        if self.cfg.row_rot_mode == "restore" and self.pack.R_out_blocks is not None:
            y_lin = apply_output_restore(y_lin, self.pack)
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
            print(
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
) -> None:
    per_layer_wbits = per_layer_wbits or {}
    replaced = 0
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
        if dry_run:
            msg = (
                f"[DUQUANT][DRYRUN] {name}: Linear({mod.in_features}->{mod.out_features}) "
                f"W{wbits} A{cfg.act_bits} perm={cfg.enable_permute} "
                f"block_in={cfg.block_size} block_out={cfg.block_out_size} row_rot={cfg.row_rot_mode}"
            )
            print(msg)
            listed += 1
            continue
        dq = DuQuantLinear(mod, name=name, cfg=cfg, weight_bits=wbits)
        setattr(parent, attr, dq)
        print(
            f"[DUQUANT][REPLACED] {name}: Linear({mod.in_features}->{mod.out_features}) -> DuQuantLinear "
            f"W{wbits} A{cfg.act_bits} perm={cfg.enable_permute} block_in={cfg.block_size} block_out={cfg.block_out_size} row_rot={cfg.row_rot_mode}"
        )
        replaced += 1
    if dry_run:
        print(f"[DUQUANT] Dry-run total layers listed: {listed}")
    else:
        print(f"[DUQUANT] Total layers replaced: {replaced}")


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
    inc = env.get("OPENPI_DUQUANT_INCLUDE", r".*(q_proj|k_proj|v_proj|out_proj|fc1|fc2|up_proj|down_proj).*")
    exc = env.get("OPENPI_DUQUANT_EXCLUDE", r"(?:^|\.)(norm|ln|layernorm|emb)(?:\.|$)")
    per_layer_wbits = _parse_per_layer_wbits(env.get("OPENPI_DUQUANT_WBITS"))
    dry_run = env.get("OPENPI_DUQUANT_DRYRUN", "0") not in ("0", "false", "False")

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
    print(f"[DUQUANT] Matched Linear layers: {len(layer_names)}")
    if dry_run:
        wrap_duquant(model, layer_names, cfg, per_layer_wbits, dry_run=True)
        return
    wrap_duquant(model, layer_names, cfg, per_layer_wbits, dry_run=False)
