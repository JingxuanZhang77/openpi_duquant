#!/usr/bin/env python
"""Collect SmoothQuant scaling factors for Pi0.5 LLM linear layers.

This script runs a short LIBERO rollout using the FP teacher model, records the
maximum absolute activation per input channel for each language-model Linear,
and computes SmoothQuant per-channel scales (activation^alpha / weight^(1-alpha)).
The resulting JSON can be loaded at inference time to rescale inputs + weights.

Usage:
    python tools/smoothquant_llm.py \
        --checkpoint ckpts/pi05_libero_torch \
        --out smoothquant_llm.json \
        --steps 32 \
        --alpha 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Iterator

import numpy as np
import torch

from examples.libero.main import LIBERO_ENV_RESOLUTION, _get_libero_env, _quat2axisangle
from libero.libero import benchmark
from openpi.policies import policy_config
from openpi.policies import policy as policy_module
from openpi.training import config as train_config
from openpi_client import image_tools

logging.basicConfig(level=logging.INFO)

TARGET_PREFIX = "paligemma_with_expert.paligemma.model.language_model"
LINEAR_SUFFIXES = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".mlp.down_proj",
)


class ActivationTracker:
    """Track per-channel maximum absolute activation for a Linear layer."""

    def __init__(self, in_features: int):
        self.in_features = in_features
        self.device: torch.device | None = None
        self.max_act: torch.Tensor | None = None

    def _ensure_buffer(self, device: torch.device) -> None:
        if self.max_act is None or self.device != device:
            self.device = device
            self.max_act = torch.zeros(self.in_features, dtype=torch.float32, device=device)

    def update(self, tensor: torch.Tensor) -> None:
        if tensor.ndim < 1:
            return
        self._ensure_buffer(tensor.device)
        last_dim = tensor.shape[-1]
        values = tensor.detach().abs().reshape(-1, last_dim)
        channel_max = values.max(dim=0)[0]
        self.max_act = torch.maximum(self.max_act, channel_max)


def _iter_observations(task_suite_name: str, steps: int, seed: int) -> Iterator[dict]:
    bench = benchmark.get_benchmark_dict()
    suite = bench[task_suite_name]()
    rng = np.random.default_rng(seed)
    for idx in range(steps):
        task_id = idx % suite.n_tasks
        task = suite.get_task(task_id)
        env, task_desc = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed + idx)
        try:
            init_states = suite.get_task_init_states(task_id)
            state = init_states[rng.integers(len(init_states))]
            obs = env.set_init_state(state)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to set init state (%s); falling back to reset()", exc)
            obs = env.reset()

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION)
        )
        wrist = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION)
        )
        state_vec = np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        )

        yield {
            "observation/image": img,
            "observation/wrist_image": wrist,
            "observation/state": state_vec,
            "prompt": str(task_desc),
        }
        env.close()


def _is_target_layer(name: str) -> bool:
    if not name.startswith(TARGET_PREFIX):
        return False
    return any(name.endswith(suffix) for suffix in LINEAR_SUFFIXES)


def _attach_trackers(model: torch.nn.Module) -> tuple[Dict[str, ActivationTracker], list]:
    trackers: Dict[str, ActivationTracker] = {}
    handles = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not _is_target_layer(name):
            continue

        tracker = trackers.setdefault(name, ActivationTracker(module.in_features))

        def hook(mod, inp, _out, tracker=tracker):
            if not inp:
                return
            tracker.update(inp[0])

        handles.append(module.register_forward_hook(hook))
    return trackers, handles


def _compute_scale(act_max: torch.Tensor, weight: torch.Tensor, alpha: float) -> torch.Tensor:
    eps = 1e-6
    act = torch.maximum(act_max.cpu(), torch.full_like(act_max.cpu(), eps))
    with torch.no_grad():
        w = torch.maximum(weight.detach().abs().amax(dim=0).cpu(), torch.full_like(act, eps))
        log_scale = alpha * torch.log(act) - (1 - alpha) * torch.log(w)
        scale = torch.exp(log_scale)
    scale = torch.clamp(scale, min=1e-6, max=1e6)
    return scale


def _load_policy(checkpoint: Path, device: str) -> policy_module.Policy:
    cfg = train_config.get_config("pi05_libero")
    with torch.device(device):
        policy = policy_config.create_trained_policy(cfg, checkpoint, pytorch_device=device)
    return policy


def smoothquant_calibrate(checkpoint: Path, steps: int, out_path: Path, alpha: float, seed: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Loading FP teacher from %s", checkpoint)
    policy = _load_policy(checkpoint, device)
    model = policy._model
    trackers, handles = _attach_trackers(model)

    logging.info("Collecting %d LIBERO observations...", steps)
    obs_iter = _iter_observations("libero_10", steps, seed)
    for idx, element in enumerate(obs_iter, start=1):
        policy.infer(element)
        if idx % 8 == 0:
            logging.info("  Processed %d/%d samples", idx, steps)

    for handle in handles:
        handle.remove()

    logging.info("Computing SmoothQuant scales (alpha=%.2f)...", alpha)
    scales: Dict[str, list[float]] = {}
    for name, tracker in trackers.items():
        module = dict(model.named_modules())[name]
        scale = _compute_scale(tracker.max_act, module.weight, alpha)
        scales[name] = scale.tolist()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"alpha": alpha, "layers": scales}
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logging.info("SmoothQuant map written to %s (%d layers)", out_path, len(scales))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmoothQuant calibration for Pi0.5 LLM")
    parser.add_argument("--checkpoint", type=Path, required=True, help="pi0.5 checkpoint directory")
    parser.add_argument("--steps", type=int, default=32, help="Number of calibration samples")
    parser.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant alpha (0-1)")
    parser.add_argument("--out", type=Path, default=Path("smoothquant_llm.json"), help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    smoothquant_calibrate(args.checkpoint.expanduser(), args.steps, args.out.expanduser(), args.alpha, args.seed)


if __name__ == "__main__":
    main()
