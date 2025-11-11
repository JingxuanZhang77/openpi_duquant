#!/usr/bin/env python
"""ATM 校准脚本 - 精简验证版本

使用方法:
    python tools/calibrate_atm_dit_v2.py --steps 16 --scope dit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import types
from pathlib import Path
import time
from typing import Dict

import numpy as np
import torch

# 路径设置
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party/libero"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Torch load 配置
_original_torch_load = torch.load
torch.load = lambda *a, **k: _original_torch_load(*a, **{**dict(k), "weights_only": False})

# Numba stub
if "numba" not in sys.modules:
    numba_stub = types.ModuleType("numba")
    numba_stub.jit = lambda *a, **k: (lambda fn: fn)  # type: ignore
    sys.modules["numba"] = numba_stub

from examples.libero.main import LIBERO_ENV_RESOLUTION, _get_libero_env, _quat2axisangle
from libero.libero import benchmark
from openpi.policies import policy_config
from openpi.training import config as train_config
from openpi.models_pytorch.atm_dit import ensure_dit_attention_patch

logging.basicConfig(level=logging.INFO)


class HeadStdTracker:
    def __init__(self, num_heads: int):
        self.sum = torch.zeros(num_heads, dtype=torch.float64)
        self.count = 0

    def update(self, values: torch.Tensor) -> None:
        if values.ndim == 1:
            values = values.unsqueeze(0)
        self.sum += values.sum(dim=0, dtype=torch.float64)
        self.count += values.shape[0]

    def mean(self) -> torch.Tensor:
        if self.count == 0:
            return torch.ones_like(self.sum, dtype=torch.float32)
        return (self.sum / self.count).to(torch.float32)


def load_model(ckpt: Path, device: str, use_duquant: bool):
    """加载模型"""
    cfg = train_config.get_config("pi05_libero")

    if use_duquant:
        # 设置 DuQuant 环境变量
        os.environ.update({
            "OPENPI_DUQUANT_WBITS_DEFAULT": "4",
            "OPENPI_DUQUANT_ABITS": "8",
            "OPENPI_DUQUANT_BLOCK": "64",
            "OPENPI_DUQUANT_PERMUTE": "1",
            "OPENPI_DUQUANT_ROW_ROT": "restore",
            "OPENPI_DUQUANT_ACT_PCT": "99.9",
            "OPENPI_DUQUANT_CALIB_STEPS": "8",  # 减少到 8 步
            "OPENPI_DUQUANT_LS": "0.15",
            "OPENPI_DUQUANT_PACKDIR": str(ckpt.parent / "duquant_packed_atm_calib"),
            "OPENPI_DUQUANT_SCOPE": "",
            "OPENPI_DUQUANT_INCLUDE": r'.*(language_model\.(.*\.)?(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|gemma_expert\.model\.layers\.\d+\.(gate_proj|up_proj|down_proj)).*',
            "OPENPI_DUQUANT_EXCLUDE": r'(?:^|\.)(vision_tower|vision|multi_modal_projector|norm|ln|layernorm|embed_tokens|lm_head)(?:\.|$)',
            "ATM_ENABLE": "0",
        })
    else:
        # 清除 DuQuant 变量
        for key in list(os.environ.keys()):
            if key.startswith("OPENPI_DUQUANT_"):
                os.environ.pop(key)
        os.environ["ATM_ENABLE"] = "0"

    policy = policy_config.create_trained_policy(cfg, ckpt, pytorch_device=device)
    return policy, policy._model


def attach_callbacks(model, trackers: Dict, scope: str):
    """设置 capture callbacks"""
    count = 0
    for name, module in model.named_modules():
        if not hasattr(module, "config"):
            continue

        # 过滤层
        if scope == "dit" and "paligemma_with_expert.gemma_expert" not in name:
            continue
        if scope == "llm" and "language_model" not in name:
            continue
        if scope == "all" and "paligemma_with_expert" not in name:
            continue

        if "self_attn" not in name:
            continue

        num_heads = module.config.num_attention_heads
        tracker = trackers.setdefault(name, HeadStdTracker(num_heads))

        # 直接设置 callback
        def make_callback(t):
            return lambda values: t.update(values)

        module._atm_capture_callback = make_callback(tracker)
        count += 1

    print(f"  已设置 {count} 个 {scope} attention 层的 callbacks")
    return count


def clear_callbacks(model):
    """清除 callbacks"""
    for module in model.modules():
        if hasattr(module, "_atm_capture_callback"):
            module._atm_capture_callback = None


def get_dummy_observation(task_id: int, seed: int):
    """获取一个测试 observation"""
    bench = benchmark.get_benchmark("libero_10")()
    task = bench.get_task(task_id)
    env = _get_libero_env(task)
    env.seed(seed)
    env.reset()

    obs = env.get_observation()
    img = obs["agentview_image"][::-1]  # RGB
    state = obs["robot0_eef_pos"].tolist() + _quat2axisangle(obs["robot0_eef_quat"]).tolist()

    env.close()

    return {
        "image": torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
        "state": torch.tensor([state], dtype=torch.float32),
        "instruction": task.language
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("~/VLM_REPO/openpi/ckpts/pi05_libero_torch"))
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--scope", choices=["dit", "llm", "all"], default="dit")
    parser.add_argument("--out", type=Path, default=Path("atm_alpha.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.ckpt = args.ckpt.expanduser()
    args.out = args.out.expanduser()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print(f"ATM 校准: {args.scope} layers, {args.steps} steps")
    print("="*80)
    print()

    # 确保 patch
    ensure_dit_attention_patch()

    # 加载模型
    print("加载 teacher 模型...")
    teacher_policy, teacher_model = load_model(args.ckpt, device, use_duquant=False)
    print("✓ Teacher 加载完成\n")

    print("加载 quant 模型...")
    quant_policy, quant_model = load_model(args.ckpt, device, use_duquant=True)
    print("✓ Quant 加载完成\n")

    # 运行校准
    teacher_trackers = {}
    quant_trackers = {}

    print(f"运行 {args.steps} 步校准...")
    print("⚠️  第一步会很慢（CUDA 初始化 + DuQuant calibration），请耐心等待\n")

    for step in range(args.steps):
        step_start = time.time()

        # 获取数据
        task_id = step % 10
        obs = get_dummy_observation(task_id, args.seed + step)

        # Teacher
        attach_callbacks(teacher_model, teacher_trackers, args.scope)
        with torch.no_grad():
            teacher_policy.infer(obs)
        clear_callbacks(teacher_model)

        # Quant
        attach_callbacks(quant_model, quant_trackers, args.scope)
        with torch.no_grad():
            quant_policy.infer(obs)
        clear_callbacks(quant_model)

        elapsed = time.time() - step_start
        print(f"[{step+1}/{args.steps}] {elapsed:.1f}s")

    print()

    # 计算 alpha
    alphas = {}
    for name in teacher_trackers:
        if name not in quant_trackers:
            continue

        teacher_std = teacher_trackers[name].mean()
        quant_std = quant_trackers[name].mean()

        alpha_raw = teacher_std / (quant_std + 1e-6)
        alpha = torch.clamp(alpha_raw, 0.8, 1.25)
        close_mask = torch.isclose(alpha, torch.ones_like(alpha), atol=0.05)
        alpha = torch.where(close_mask, torch.ones_like(alpha), alpha)

        alphas[name] = {"all": alpha.tolist()}

        print(f"{name}:")
        print(f"  teacher_std: {teacher_std.numpy()}")
        print(f"  quant_std: {quant_std.numpy()}")
        print(f"  alpha: {alpha.numpy()}")

    # 保存
    with open(args.out, "w") as f:
        json.dump(alphas, f, indent=2)

    print(f"\n✅ 保存到: {args.out}")


if __name__ == "__main__":
    main()
