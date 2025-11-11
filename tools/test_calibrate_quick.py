#!/usr/bin/env python
"""快速测试：验证校准脚本能否正确加载两个模型"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party/libero"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ.pop("OPENPI_DUQUANT_WBITS_DEFAULT", None)  # 清除可能存在的变量

import torch
from openpi.models_pytorch.atm_dit import ensure_dit_attention_patch
from openpi.policies import policy_config
from openpi.training import config as train_config

# 导入校准脚本的函数
import importlib.util
spec = importlib.util.spec_from_file_location("calibrate", Path(__file__).parent / "calibrate_atm_dit.py")
calibrate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate_module)

print("="*80)
print("测试校准脚本")
print("="*80)
print()

ckpt = Path("~/VLM_REPO/openpi/ckpts/pi05_libero_torch").expanduser()
device = "cuda" if torch.cuda.is_available() else "cpu"

ensure_dit_attention_patch()

print("1. 测试加载 Teacher 模型（应该是 FP16）...")
teacher_policy, teacher_model = calibrate_module._policy_from_checkpoint(ckpt, device, duquant_enabled=False)
print("   ✅ Teacher 加载成功")

# 检查是否有 DuQuant 层
has_duquant = any("DuQuant" in m.__class__.__name__ for m in teacher_model.modules())
if has_duquant:
    print("   ❌ 错误：Teacher 不应该有 DuQuant 层！")
    sys.exit(1)
else:
    print("   ✅ Teacher 没有 DuQuant 层（正确）")

print()
print("2. 测试加载 Quant 模型（应该是 W4A8）...")
quant_policy, quant_model = calibrate_module._policy_from_checkpoint(ckpt, device, duquant_enabled=True)
print("   ✅ Quant 加载成功")

# 检查是否有 DuQuant 层
duquant_layers = [name for name, m in quant_model.named_modules() if "DuQuant" in m.__class__.__name__]
if not duquant_layers:
    print("   ❌ 错误：Quant 应该有 DuQuant 层！")
    print("   可能原因：DuQuant 环境变量没有正确设置")
    sys.exit(1)
else:
    print(f"   ✅ Quant 有 {len(duquant_layers)} 个 DuQuant 层（正确）")
    print(f"   示例: {duquant_layers[0]}")

print()
print("3. 检查 DiT attention 层...")
dit_layers = []
for name, module in quant_model.named_modules():
    if "paligemma_with_expert.gemma_expert" in name and "self_attn" in name:
        if hasattr(module, "config"):
            dit_layers.append(name)

if not dit_layers:
    print("   ❌ 错误：没有找到 DiT attention 层！")
    sys.exit(1)
else:
    print(f"   ✅ 找到 {len(dit_layers)} 个 DiT attention 层")
    print(f"   示例: {dit_layers[0]}")

print()
print("="*80)
print("✅ 所有测试通过！校准脚本应该能正常工作")
print("="*80)
print()
print("运行完整校准:")
print("  bash tools/run_atm_simple.sh")
print()
print("快速测试（16 步）:")
print("  STEPS=16 bash tools/run_atm_simple.sh")
