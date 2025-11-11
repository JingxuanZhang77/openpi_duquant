#!/usr/bin/env python
"""最简单的 ATM 测试 - 只运行 1 步验证逻辑"""

import sys
import os
from pathlib import Path

# 设置路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party/libero"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from openpi.models_pytorch.atm_dit import ensure_dit_attention_patch
from openpi.policies import policy_config
from openpi.training import config as train_config

print("="*80)
print("测试 1: 检查 attention patch")
print("="*80)
ensure_dit_attention_patch()
print("✓ Attention patch 已应用\n")

print("="*80)
print("测试 2: 加载模型并查找 DiT attention 层")
print("="*80)

# 清除所有 DuQuant 变量
for key in list(os.environ.keys()):
    if key.startswith("OPENPI_DUQUANT_") or key.startswith("ATM_"):
        os.environ.pop(key)

cfg = train_config.get_config("pi05_libero")
ckpt = Path("~/VLM_REPO/openpi/ckpts/pi05_libero_torch").expanduser()
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"加载模型从: {ckpt}")
print(f"设备: {device}")

policy = policy_config.create_trained_policy(cfg, ckpt, pytorch_device=device)
model = policy._model

print("\n查找 DiT attention 层:")
dit_attention_layers = []
for name, module in model.named_modules():
    if "paligemma_with_expert.gemma_expert" in name and "self_attn" in name:
        if hasattr(module, "config"):
            dit_attention_layers.append(name)
            num_heads = module.config.num_attention_heads
            print(f"  ✓ {name} (num_heads={num_heads})")

if not dit_attention_layers:
    print("  ❌ 没有找到 DiT attention 层！")
    sys.exit(1)
else:
    print(f"\n✓ 找到 {len(dit_attention_layers)} 个 DiT attention 层")

print("\n="*80)
print("测试 3: 设置 capture callback")
print("="*80)

class SimpleTracker:
    def __init__(self):
        self.called = False
        self.values = []

    def callback(self, values):
        self.called = True
        self.values.append(values.shape)
        print(f"  ✓ Callback 被调用! values.shape = {values.shape}")

tracker = SimpleTracker()

for name, module in model.named_modules():
    if name in dit_attention_layers:
        module._atm_capture_callback = tracker.callback
        print(f"  设置 callback 到: {name}")

print("\n="*80)
print("测试 4: 运行一次推理")
print("="*80)

# 创建假数据
from examples.libero.main import LIBERO_ENV_RESOLUTION
from openpi_client import image_tools

dummy_obs = {
    "image": torch.rand(1, 3, LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION),
    "state": torch.rand(1, 7),
    "instruction": "pick up the object"
}

print("运行推理...")
with torch.no_grad():
    try:
        policy.infer(dummy_obs)
    except Exception as e:
        print(f"推理出错（可能正常）: {e}")

print("\n="*80)
print("结果")
print("="*80)

if tracker.called:
    print(f"✅ Callback 被调用了 {len(tracker.values)} 次")
    print(f"   Shapes: {tracker.values[:5]}")  # 显示前5个
else:
    print("❌ Callback 没有被调用!")
    print("   可能原因:")
    print("   1. patched attention 没有生效")
    print("   2. 层名不匹配")
    print("   3. callback 设置不对")

print("\n完成!")
