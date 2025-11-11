#!/usr/bin/env python
"""Calibrate DiT Attention Temperature Matching (ATM) coefficients.

The script constructs both teacher (FP16/BF16) and DuQuant W4A8 versions of the
Pi0.5 action expert, runs a small Libero roll-out to gather attention logits,
and emits per-head alpha values that can be loaded at inference time.

Usage example::

    python tools/calibrate_atm_dit.py \
        --teacher-checkpoint ~/ckpts/pi05_libero_torch \
        --quant-checkpoint   ~/ckpts/pi05_libero_torch \
        --steps 256 \
        --out atm_alpha_dit.json

The checkpoints can be identical: the teacher run ignores DuQuant env vars,
whereas the quant run honours the caller's DuQuant configuration.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import types
from pathlib import Path
import random
from typing import Any, Dict, Iterator, Tuple
import copy

import numpy as np
import torch
from torch.serialization import add_safe_globals

# Allow legacy numpy objects in LIBERO init-state checkpoints (PyTorch >=2.6 safety).
try:
    add_safe_globals([np.core.multiarray._reconstruct, np.ndarray])  # type: ignore[attr-defined]
except Exception:
    pass

_original_torch_load = torch.load


def _torch_load_with_weights(flagged_args, flagged_kwargs):
    kwargs = dict(flagged_kwargs)
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*flagged_args, **kwargs)


torch.load = lambda *a, **k: _torch_load_with_weights(a, k)  # noqa: E731

_DATASETS_ROOT = Path(__file__).resolve().parent.parent / "examples" / "libero" / "dataset" / "datasets"
_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

if "numba" not in sys.modules:
    numba_stub = types.ModuleType("numba")

    def _jit(*args, **kwargs):
        def _inner(fn):
            return fn

        return _inner

    numba_stub.jit = _jit  # type: ignore[attr-defined]
    sys.modules["numba"] = numba_stub

from examples.libero.main import LIBERO_ENV_RESOLUTION, _get_libero_env, _quat2axisangle
from libero.libero import benchmark
from openpi.policies import policy_config
from openpi.policies import policy as policy_module
from openpi.training import config as train_config
from openpi_client import image_tools
from openpi.models_pytorch.atm_dit import ensure_dit_attention_patch


logging.basicConfig(level=logging.INFO)


class HeadStdTracker:
    """Accumulate per-head standard deviation statistics.

    >>> tracker = HeadStdTracker(2)
    >>> tracker.update(torch.tensor([[1.0, 2.0],[3.0, 4.0]]))
    >>> tracker.mean().tolist()
    [2.0, 3.0]
    """

    def __init__(self, num_heads: int):
        self.sum = torch.zeros(num_heads, dtype=torch.float64)
        self.count = 0

    def update(self, values: torch.Tensor) -> None:
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2:
            raise ValueError("expected [batch, heads] std tensor")
        self.sum += values.sum(dim=0, dtype=torch.float64)
        self.count += values.shape[0]

    def mean(self) -> torch.Tensor:
        if self.count == 0:
            return torch.ones_like(self.sum, dtype=torch.float32)
        return (self.sum / self.count).to(torch.float32)


@contextlib.contextmanager
def _temporary_env_clear(prefix: str) -> Iterator[None]:
    backup = {k: os.environ.pop(k) for k in list(os.environ.keys()) if k.startswith(prefix)}
    try:
        yield
    finally:
        os.environ.update(backup)


@contextlib.contextmanager
def _temporary_env_override(key: str, value: str) -> Iterator[None]:
    had_key = key in os.environ
    old_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if had_key and old_value is not None:
            os.environ[key] = old_value
        else:
            os.environ.pop(key, None)


def _iter_observations(task_suite_name: str, steps: int, seed: int) -> Iterator[dict]:
    bench = benchmark.get_benchmark_dict()
    suite = bench[task_suite_name]()
    rng = random.Random(seed)
    for idx in range(steps):
        task_id = idx % suite.n_tasks
        task = suite.get_task(task_id)
        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed + idx)
        try:
            init_states = suite.get_task_init_states(task_id)
            chosen = init_states[rng.randrange(len(init_states))]
            obs = env.set_init_state(chosen)
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "ATM calibration: failed to load init state for task %s (reason: %s); falling back to env.reset().",
                task_id,
                exc,
            )
            try:
                obs = env.reset()
            except Exception as reset_exc:  # noqa: BLE001
                logging.error(
                    "ATM calibration: env.reset() also failed for task %s (reason: %s); skipping sample.",
                    task_id,
                    reset_exc,
                )
                env.close()
                continue

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION))
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION)
        )
        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )

        yield {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            "prompt": str(task.language),
        }
        env.close()


def _attach_capture(model: torch.nn.Module, trackers: Dict[str, HeadStdTracker]) -> None:
    from transformers.models.gemma import modeling_gemma as hf_gemma

    for name, module in model.named_modules():
        if not isinstance(module, hf_gemma.GemmaAttention):
            continue
        if "paligemma_with_expert.gemma_expert" not in name:
            continue
        if not hasattr(module, "_atm_capture_callback"):
            module._atm_capture_callback = None  # type: ignore[attr-defined]

        num_heads = module.config.num_attention_heads
        tracker = trackers.setdefault(name, HeadStdTracker(num_heads))

        def _callback_factory(t: HeadStdTracker) -> callable:
            def _callback(values: torch.Tensor) -> None:
                t.update(values.to(torch.float32))

            return _callback

        module._atm_capture_callback = _callback_factory(tracker)


def _clear_capture(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "_atm_capture_callback"):
            module._atm_capture_callback = None


def _policy_from_checkpoint(checkpoint: Path, device: str, duquant_enabled: bool) -> Tuple[policy_module.Policy, torch.nn.Module]:
    cfg = train_config.get_config("pi05_libero")
    target_device = device if device != "cuda" else "cpu"
    if duquant_enabled:
        default_packdir = os.environ.get(
            "OPENPI_DUQUANT_PACKDIR",
            str(Path(os.environ.get("OPENPI_DUQUANT_PACKROOT", checkpoint.parent)) / "/home/jz97/VLM_REPO/openpi/duquant_packed_llm_w4a8_atm"),
        )
        duquant_defaults = {
            "OPENPI_DUQUANT_WBITS_DEFAULT": "4",
            "OPENPI_DUQUANT_ABITS": "4",
            "OPENPI_DUQUANT_BLOCK": "64",
            "OPENPI_DUQUANT_PERMUTE": "1",
            "OPENPI_DUQUANT_ROW_ROT": "restore",
            "OPENPI_DUQUANT_ACT_PCT": "99.9",
            "OPENPI_DUQUANT_CALIB_STEPS": os.environ.get("OPENPI_DUQUANT_CALIB_STEPS", "32"),
            "OPENPI_DUQUANT_LS": "0.15",
            "OPENPI_DUQUANT_PACKDIR": default_packdir,
            "OPENPI_DUQUANT_SCOPE": "",
            "OPENPI_DUQUANT_INCLUDE": (
    r'.*(paligemma_with_expert\.paligemma\.model\.language_model'
    r'\.(?:.*\.)?(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'
    r'|paligemma_with_expert\.gemma_expert\.model'
    r'\.(?:layers\.\d+\.)?mlp\.(gate_proj|up_proj|down_proj)).*'
),
            "OPENPI_DUQUANT_EXCLUDE": r'(?:^|\.)(vision_tower|vision|multi_modal_projector|norm|ln|layernorm|embed_tokens|lm_head)(?:\.|$)',
            "ATM_ENABLE": "0",
        }



        backup: Dict[str, str] = {}
        applied: list[str] = []
        for key, value in duquant_defaults.items():
            if key in os.environ:
                backup[key] = os.environ[key]
            else:
                os.environ[key] = value
                applied.append(key)

        try:
            policy = policy_config.create_trained_policy(cfg, checkpoint, pytorch_device=target_device)
        finally:
            for key in applied:
                os.environ.pop(key, None)
            for key, value in backup.items():
                os.environ[key] = value
        policy._pytorch_device = target_device
    else:
        # Teacher model: 清除所有 DuQuant 变量
        with _temporary_env_clear("OPENPI_DUQUANT_"):
            with _temporary_env_override("ATM_ENABLE", "0"):
                policy = policy_config.create_trained_policy(cfg, checkpoint, pytorch_device=target_device)
        policy._pytorch_device = target_device

    return policy, policy._model


def _run_policy(policy: policy_module.Policy, element: dict) -> None:
    policy.infer(element)


def calibrate(teacher_ckpt: Path, quant_ckpt: Path, steps: int, out_path: Path, seed: int) -> None:
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("加载模型...")
    print("=" * 80)

    ensure_dit_attention_patch()

    print("加载 Teacher 模型（FP16）...")
    teacher_policy, teacher_model = _policy_from_checkpoint(teacher_ckpt, device, duquant_enabled=False)
    if device == "cuda":
        teacher_policy._model.to(device)
    teacher_policy._pytorch_device = device
    print("✓ Teacher 加载完成\n")

    print("加载 Quant 模型（DuQuant W4A8）...")
    quant_policy, quant_model = _policy_from_checkpoint(quant_ckpt, device, duquant_enabled=True)
    if device == "cuda":
        quant_policy._model.to("cpu")
        torch.cuda.empty_cache()
    quant_policy._pytorch_device = "cpu" if device == "cuda" else device
    print("✓ Quant 加载完成\n")

    teacher_trackers: Dict[str, HeadStdTracker] = {}
    quant_trackers: Dict[str, HeadStdTracker] = {}

    print("=" * 80)
    print(f"开始校准（{steps} 步）")
    print("=" * 80)
    print("⚠️  首次运行若需要 DuQuant 激活校准会略慢\n")

    obs_iter = _iter_observations("libero_10", steps, seed)
    cached_elements: list[dict] = []

    for idx, element in enumerate(obs_iter):
        start = time.time()
        cached_elements.append(copy.deepcopy(element))

        print(f"[{idx + 1}/{steps}] Teacher...", end="", flush=True)
        _attach_capture(teacher_model, teacher_trackers)
        _run_policy(teacher_policy, element)
        _clear_capture(teacher_model)
        teacher_time = time.time() - start
        print(f" {teacher_time:.1f}s", flush=True)

    if device == "cuda":
        teacher_policy._model.to("cpu")
        teacher_policy._pytorch_device = "cpu"
        torch.cuda.empty_cache()
        quant_policy._model.to(device)
        quant_policy._pytorch_device = device

    for idx, element in enumerate(cached_elements):
        start = time.time()
        print(f"[{idx + 1}/{steps}] Quant...", end="", flush=True)
        _attach_capture(quant_model, quant_trackers)
        _run_policy(quant_policy, element)
        _clear_capture(quant_model)
        quant_time = time.time() - start
        print(f" {quant_time:.1f}s", flush=True)

    if device == "cuda":
        quant_policy._model.to("cpu")
        quant_policy._pytorch_device = "cpu"
        torch.cuda.empty_cache()

    logging.info("Teacher tracker keys: %s", sorted(teacher_trackers.keys()))
    logging.info("Quant tracker keys: %s", sorted(quant_trackers.keys()))

    alphas: Dict[str, Dict[str, list[float]]] = {}
    for layer_name, teacher_tracker in teacher_trackers.items():
        if layer_name not in quant_trackers:
            continue
        teacher_std = teacher_tracker.mean()
        quant_std = quant_trackers[layer_name].mean()
        alpha = torch.clamp(teacher_std / (quant_std + 1e-6), 0.8, 1.25)
        close_mask = torch.isclose(alpha, torch.ones_like(alpha), atol=0.05)
        alpha = torch.where(close_mask, torch.ones_like(alpha), alpha)
        alphas[layer_name] = {"all": alpha.tolist()}

    if not alphas:
        raise RuntimeError("ATM calibration collected no statistics; ensure benchmarks are accessible")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(alphas, fh, indent=2)
    print(f"Wrote ATM alpha map to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate DiT attention temperature matching")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--quant-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("atm_alpha_dit.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibrate(args.teacher_checkpoint.expanduser(), args.quant_checkpoint.expanduser(), args.steps, args.out.expanduser(), args.seed)


if __name__ == "__main__":
    main()
