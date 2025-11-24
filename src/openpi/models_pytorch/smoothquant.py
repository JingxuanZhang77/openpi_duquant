"""Runtime helpers for applying SmoothQuant scaling to Pi0.5 LLM."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import torch

try:
    from openpi.models_pytorch.duquant_layers import DuQuantLinear
except Exception:  # pragma: no cover - DuQuant not available
    DuQuantLinear = tuple()  # type: ignore[assignment]

_TARGET_PREFIX = "paligemma_with_expert.paligemma.model.language_model"


def _is_target(name: str) -> bool:
    return name.startswith(_TARGET_PREFIX)


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def enable_llm_smoothquant_if_configured(model: torch.nn.Module) -> None:
    """Apply SmoothQuant scaling if SMOOTHQUANT_ENABLE is set."""

    if not _truthy(os.getenv("SMOOTHQUANT_ENABLE")):
        return

    path = os.getenv("SMOOTHQUANT_ALPHA_PATH")
    if not path:
        print("SmoothQuant skipped: SMOOTHQUANT_ALPHA_PATH not set")
        return

    scale_path = Path(path).expanduser()
    if not scale_path.exists():
        print(f"SmoothQuant skipped: file not found at {scale_path}")
        return

    with scale_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    layer_scales: Dict[str, list[float]] = payload.get("layers", payload)

    applied = 0
    linear_types = (torch.nn.Linear,)
    if isinstance(DuQuantLinear, type):
        linear_types = (torch.nn.Linear, DuQuantLinear)

    for name, module in model.named_modules():
        if not isinstance(module, linear_types):
            continue
        if not _is_target(name):
            continue
        scale_vals = layer_scales.get(name)
        if not scale_vals:
            continue
        scale = torch.tensor(scale_vals, dtype=module.weight.dtype, device=module.weight.device)
        if scale.shape[0] != module.in_features:
            print(f"SmoothQuant warning: scale size mismatch for {name}")
            continue
        if getattr(module, "_smoothquant_applied", False):
            continue

        with torch.no_grad():
            module.weight.div_(scale)
        scale = scale.reshape(1, -1)

        def _pre_hook(mod, inputs, _scale=scale):
            if not inputs:
                return inputs
            x = inputs[0]
            reshaped = _scale.to(x.device, x.dtype)
            x = x * reshaped
            return (x,)

        module.register_forward_pre_hook(_pre_hook, with_kwargs=False)
        module._smoothquant_applied = True  # type: ignore[attr-defined]
        applied += 1

    if applied:
        print(f"SmoothQuant enabled: applied scales to {applied} LLM Linear layers (source={scale_path})")
    else:
        print("SmoothQuant warning: no LLM layers matched scale map")
