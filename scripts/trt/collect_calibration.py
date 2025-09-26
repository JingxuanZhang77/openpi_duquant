#!/usr/bin/env python3
"""Utility to dump calibration batches for TensorRT INT8 builds."""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))

try:
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    missing = exc.name or "dependency"
    raise SystemExit(
        f"Missing dependency '{missing}'. Install the OpenPI package and its requirements before running this script."
    ) from exc


def _random_observation(
    *,
    image_size: tuple[int, int],
    state_dim: int,
    prompt: str,
) -> dict:
    height, width = image_size
    return {
        "observation/image": np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8),
        "observation/state": np.random.randn(state_dim).astype(np.float32),
        "prompt": prompt,
    }


def _prune_none(tree):
    if isinstance(tree, dict):
        return {k: _prune_none(v) for k, v in tree.items() if v is not None}
    return tree


def collect_batches(args: argparse.Namespace) -> None:
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.duquant_scope:
        os.environ.setdefault("OPENPI_DUQUANT_SCOPE", args.duquant_scope)

    if args.disable_torch_compile is not None:
        os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", str(int(args.disable_torch_compile)))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        pytorch_device=args.device,
    )
    model = policy._model
    model.eval()

    for batch_idx in range(args.num_batches):
        samples = []
        raw_prompts: list[str] = []
        for _ in range(args.batch_size):
            sample = _random_observation(
                image_size=(args.image_height, args.image_width),
                state_dim=args.state_dim,
                prompt=args.prompt,
            )
            raw_prompts.append(str(sample["prompt"]))
            processed = policy._input_transform(sample)
            samples.append(_prune_none(processed))

        collated = default_collate(samples)
        payload: dict[str, object] = {
            "images": collated.get("image"),
            "image_masks": collated.get("image_mask"),
            "states": collated.get("state"),
            "tokenized_prompts": collated.get("tokenized_prompt"),
            "tokenized_prompt_masks": collated.get("tokenized_prompt_mask"),
            "raw_prompts": raw_prompts,
        }
        if "token_ar_mask" in collated:
            payload["token_ar_mask"] = collated["token_ar_mask"]
        if "token_loss_mask" in collated:
            payload["token_loss_mask"] = collated["token_loss_mask"]

        torch.save(payload, out_dir / f"batch_{batch_idx:04d}.pt")
        print(f"[calib] dump batch {batch_idx}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Directory containing the pi0.5_LIBERO checkpoint")
    parser.add_argument("--config", default="pi05_libero", help="Training config to load (default: pi05_libero)")
    parser.add_argument("--out", default="calibration_batches", help="Directory to store calibration batches")
    parser.add_argument("--num-batches", type=int, default=64, help="Number of batches to dump")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per calibration batch")
    parser.add_argument("--state-dim", type=int, default=8, help="Dimension of the proprioceptive state vector")
    parser.add_argument("--image-height", type=int, default=224, help="Image height in pixels")
    parser.add_argument("--image-width", type=int, default=224, help="Image width in pixels")
    parser.add_argument("--prompt", default="pick up the object", help="Prompt text for synthetic samples")
    parser.add_argument("--device", default="cuda", help="Device to load the PyTorch policy onto")
    parser.add_argument("--duquant-scope", default=None, help="Override OPENPI_DUQUANT_SCOPE if provided")
    parser.add_argument(
        "--disable-torch-compile",
        type=int,
        choices=(0, 1),
        default=None,
        help="If set, controls OPENPI_DISABLE_TORCH_COMPILE (default: leave unchanged)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for RNGs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_batches(args)


if __name__ == "__main__":
    main()
