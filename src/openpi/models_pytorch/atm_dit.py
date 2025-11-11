"""Utilities for enabling DiT-specific Attention Temperature Matching (ATM).

This module reads per-head alpha coefficients from a JSON file and binds them to
the Gemma attention layers that comprise the Pi0.5 action expert (DiT).  At
runtime the attention implementation multiplies each head's query tensor by the
stored alpha, mitigating quantisation-induced temperature drift.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

from transformers.models.gemma import modeling_gemma as hf_gemma
from openpi.models_pytorch.transformers_replace.models.gemma import modeling_gemma as patched_gemma


def _ensure_gemma_attention_patch() -> None:
    if getattr(hf_gemma.GemmaAttention, "_atm_patch_applied", False):
        return

    original_cls = hf_gemma.GemmaAttention

    class _ATMGemmaAttention(patched_gemma.GemmaAttention):  # type: ignore[misc]
        """ATM-aware Gemma attention that mirrors the HF implementation."""

    _ATMGemmaAttention.__name__ = original_cls.__name__
    _ATMGemmaAttention.__qualname__ = original_cls.__qualname__
    _ATMGemmaAttention.__module__ = original_cls.__module__
    _ATMGemmaAttention.__doc__ = original_cls.__doc__
    _ATMGemmaAttention._atm_patch_applied = True  # type: ignore[attr-defined]

    hf_gemma.GemmaAttention = _ATMGemmaAttention  # type: ignore[assignment]


def ensure_dit_attention_patch() -> None:
    """Ensure Gemma attention uses the ATM-aware implementation."""

    _ensure_gemma_attention_patch()


_DIT_MARKER = "paligemma_with_expert.gemma_expert"
_ATM_BOUND_ATTR = "_atm_alpha_all"


def _normalise_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def _load_alpha_json(path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    alpha_map: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_name, entry in payload.items():
        converted: Dict[str, torch.Tensor] = {}
        for key, values in entry.items():
            if key not in {"self", "cross", "all"}:
                continue
            converted[key] = torch.tensor(values, dtype=torch.float32)
        if converted:
            alpha_map[layer_name] = converted
    return alpha_map


def _select_alpha(entry: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    if "all" in entry:
        return entry["all"]
    if "self" in entry:
        return entry["self"]
    return entry.get("cross")


def enable_dit_atm_if_configured(model: torch.nn.Module) -> None:
    """Enable DiT-only ATM if the relevant environment variables are set.

    The routine is intentionally conservative: it only attaches alpha tensors
    to Gemma attention layers whose qualified name contains the DiT marker and
    appears in the provided JSON map.  LLM / VLM attention remains untouched.
    """

    if not _normalise_bool(os.getenv("ATM_ENABLE")):
        return

    scope = os.getenv("ATM_SCOPE", "dit").lower()
    if scope not in {"dit", "all"}:
        return

    alpha_path_env = os.getenv("ATM_ALPHA_PATH")
    if not alpha_path_env:
        print("ATM(DiT) skipped: ATM_ALPHA_PATH is not set")
        return

    alpha_path = Path(alpha_path_env).expanduser()
    if not alpha_path.exists():
        print(f"ATM(DiT) skipped: alpha file not found at {alpha_path}")
        return

    _ensure_gemma_attention_patch()

    alpha_map = _load_alpha_json(alpha_path)

    for module in model.modules():
        if isinstance(module, hf_gemma.GemmaAttention):
            if not hasattr(module, _ATM_BOUND_ATTR):
                setattr(module, _ATM_BOUND_ATTR, None)
            if not hasattr(module, "_atm_capture_callback"):
                module._atm_capture_callback = None  # type: ignore[attr-defined]
    matched_layers: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if not isinstance(module, hf_gemma.GemmaAttention):
            continue
        if _DIT_MARKER not in name and scope == "dit":
            continue

        entry = alpha_map.get(name)
        if not entry:
            continue

        alpha = _select_alpha(entry)
        if alpha is None:
            continue

        num_heads = module.config.num_attention_heads
        if alpha.shape[0] != num_heads:
            print(
                f"ATM(DiT) warning: head mismatch for {name} (alpha={alpha.shape[0]}, model={num_heads})"
            )
            continue

        module._atm_alpha_all = alpha.clone().detach()
        matched_layers[name] = module._atm_alpha_all

    if matched_layers:
        head_counts = sorted({tensor.numel() for tensor in matched_layers.values()})
        print(
            f"ATM(DiT) enabled: matched_layers={len(matched_layers)}, heads={head_counts}, source={alpha_path}"
        )
        for entry in sorted(matched_layers):
            print(f"  • {entry}")
    else:
        print("ATM(DiT) warning: no attention layers matched alpha map")
