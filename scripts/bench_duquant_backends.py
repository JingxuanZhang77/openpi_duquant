#!/usr/bin/env python3
import os
import time
from pathlib import Path

import torch

from openpi.policies import policy_config
from openpi.training import config as train_config


def run_once(backend: str | None, device: str = "cuda", return_output: bool = False):
    if backend:
        os.environ["OPENPI_DUQUANT_BACKEND"] = backend
    else:
        os.environ.pop("OPENPI_DUQUANT_BACKEND", None)

    # Keep the rest of DuQuant config the same (W4A8 on LLM by default)
    os.environ.setdefault("OPENPI_DUQUANT_SCOPE", "paligemma_with_expert.paligemma.model.language_model.")
    os.environ.setdefault("OPENPI_DUQUANT_WBITS_DEFAULT", "4")
    os.environ.setdefault("OPENPI_DUQUANT_ABITS", "8")
    os.environ.setdefault("OPENPI_DUQUANT_PERMUTE", "1")
    os.environ.setdefault("OPENPI_DUQUANT_ROW_ROT", "restore")

    ckpt = Path(os.environ.get("CKPT", "~/VLM_REPO/openpi/ckpts/pi05_libero_torch")).expanduser()
    policy = policy_config.create_trained_policy(
        train_config.get_config("pi05_libero"), ckpt, pytorch_device=device
    )
    model = policy._model
    model.eval()

    # Build a dummy batch that exercises the LLM path
    B = 2
    T = 16
    hidden = 2048
    x = torch.randn(B, T, hidden, device=device, dtype=torch.float16)
    # Find a target linear layer under LLM (prefer DuQuantLinear if present)
    target_name, target = None, None
    for n, m in model.named_modules():
        if n.startswith("paligemma_with_expert.paligemma.model.language_model.") and (
            m.__class__.__name__ in ("DuQuantLinear",)
        ):
            target_name, target = n, m
            break
    if target is None:
        for n, m in model.named_modules():
            if n.startswith("paligemma_with_expert.paligemma.model.language_model.") and isinstance(m, torch.nn.Linear):
                target_name, target = n, m
                break
    assert target is not None and target_name is not None

    # Warmup
    x2d = x.view(-1, hidden)
    with torch.no_grad():
        for _ in range(10):
            _ = target(x2d)

    # Timed run
    iters = 50
    torch.cuda.synchronize(device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        last = None
        for _ in range(iters):
            last = target(x2d)
    torch.cuda.synchronize(device=device)
    dt = time.perf_counter() - t0
    avg_ms = (dt / iters) * 1000.0
    print(f"Backend={backend or 'fake'} layer={target_name} avg_ms={avg_ms:.3f}")
    return (avg_ms, last.detach().float().cpu()) if return_output else avg_ms


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Fake path (unset backend)
    t_fake, y_fake = run_once(None, device=device, return_output=True)
    # BitBLAS path
    t_bitblas, y_bb = run_once("bitblas", device=device, return_output=True)
    print(f"Speedup (fake/bitblas): {t_fake / t_bitblas if t_bitblas > 0 else float('inf'):.2f}x")
    import torch
    if y_bb is not None and y_fake is not None:
        diff = (y_bb - y_fake).abs()
        rel = diff / (y_fake.abs() + 1e-6)
        print(f"Accuracy: max_abs={diff.max().item():.4e}, mean_abs={diff.mean().item():.4e}, max_rel={rel.max().item():.4e}")
