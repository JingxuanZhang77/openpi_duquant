#!/usr/bin/env python3
"""Inspect cached DuQuant weights to verify integer quantization."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch

from openpi.policies import policy_config
from openpi.training import config as train_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that DuQuantLinear buffers contain quantized integer weights.",
    )
    parser.add_argument(
        "--policy-config",
        default="pi05_libero",
        help="Training config key used to construct the policy (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("CKPT", ""),
        help="Path to the policy checkpoint directory (defaults to $CKPT).",
    )
    parser.add_argument(
        "--layer-name",
        help="Exact fully-qualified module name to inspect (takes precedence over --pattern).",
    )
    parser.add_argument(
        "--pattern",
        default=r"\.mlp\.up_proj$",
        help="Regex used to choose a DuQuantLinear layer when --layer-name is not provided.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=8,
        help="Number of output rows to sample when checking quantization (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="Device to load the PyTorch policy on; 'auto' picks cuda if available (default: cpu).",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=1,
        help="How many matched modules to report (default: %(default)s).",
    )
    return parser.parse_args()


def _select_modules(model: torch.nn.Module, layer_name: str | None, pattern: str) -> list[tuple[str, torch.nn.Module]]:
    modules = [
        (name, mod)
        for name, mod in model.named_modules()
        if mod.__class__.__name__ == "DuQuantLinear"
    ]
    if not modules:
        raise RuntimeError("No DuQuantLinear modules were found. Confirm DuQuant is enabled via environment vars.")

    if layer_name:
        for name, mod in modules:
            if name == layer_name:
                return [(name, mod)]
        available = "\n".join(name for name, _ in modules)
        raise RuntimeError(f"Layer '{layer_name}' not found among DuQuantLinear modules:\n{available}")

    regex = re.compile(pattern)
    matches = [(name, mod) for name, mod in modules if regex.search(name)]
    if not matches:
        available = "\n".join(name for name, _ in modules[:20])
        raise RuntimeError(
            "No DuQuantLinear modules matched pattern '{}'\n"
            "Example available names (first 20):\n{}".format(pattern, available)
        )
    return matches


def _pick_device(device_flag: str) -> str:
    if device_flag != "auto":
        return device_flag
    return "cuda" if torch.cuda.is_available() else "cpu"


def _print_report(name: str, module: torch.nn.Module, rows: int) -> None:
    rows = max(1, min(rows, module.out_features))
    module._maybe_update_weight_cache()  # type: ignore[attr-defined]

    Wq = getattr(module, "_W_t_quantized", None)
    scales = getattr(module, "_w_scales", None)
    if Wq is None or scales is None:
        raise RuntimeError(f"Module '{name}' is missing cached quantized weight buffers.")

    with torch.no_grad():
        Wq_rows = Wq[:rows].to(torch.float32)
        scales_rows = scales[:rows].to(torch.float32)
        scale_matrix = scales_rows.view(-1, 1)
        quant_values = Wq_rows / (scale_matrix + 1e-12)
        integers = torch.round(quant_values)
        fractional = (quant_values - integers).abs().max().item()
        residual = (Wq_rows - integers * scale_matrix).abs().max().item()
        int_min = -(2 ** (int(module.weight_bits) - 1))
        int_max = (2 ** (int(module.weight_bits) - 1)) - 1

    print(f"Layer: {name}")
    print(f"  rows checked        : {rows} / {Wq.shape[0]}")
    print(f"  cols per row        : {Wq.shape[1]}")
    print(f"  integer range       : {integers.min().item()} to {integers.max().item()} (expected within [{int_min}, {int_max}])")
    print(f"  max |fractional part|: {fractional:.6e}")
    print(f"  max |residual|       : {residual:.6e}")
    print()


def main() -> None:
    args = _parse_args()
    ckpt = Path(args.checkpoint).expanduser()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")

    device = _pick_device(args.device)

    policy = policy_config.create_trained_policy(
        train_config.get_config(args.policy_config),
        ckpt,
        pytorch_device=device,
    )
    model = policy._model
    model.eval()

    matches = _select_modules(model, args.layer_name, args.pattern)

    for name, module in matches[: args.max_layers]:
        _print_report(name, module, args.rows)


if __name__ == "__main__":
    main()
