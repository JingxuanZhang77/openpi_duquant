#!/usr/bin/env python3
"""
验证 ATM 校准时 DuQuant 是否真的生效

Usage:
    # 不设置 DuQuant 环境变量（模拟错误情况）
    python tools/verify_duquant_in_calibration.py

    # 设置 DuQuant 环境变量（模拟正确情况）
    export OPENPI_DUQUANT_WBITS_DEFAULT=4
    export OPENPI_DUQUANT_ABITS=8
    python tools/verify_duquant_in_calibration.py
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("="*80)
print("验证 ATM 校准时的 DuQuant 状态")
print("="*80)
print()

# 检查当前环境变量
print("当前 DuQuant 环境变量:")
duquant_vars = {k: v for k, v in os.environ.items() if 'DUQUANT' in k}
if duquant_vars:
    for k, v in sorted(duquant_vars.items()):
        print(f"  {k}={v}")
else:
    print("  ❌ 没有设置任何 OPENPI_DUQUANT_* 环境变量")
    print("  ⚠️  这意味着 quant_model 不会被量化！")

print()
print("-"*80)
print()

# 模拟 _policy_from_checkpoint 的行为
from tools.calibrate_atm_dit import _temporary_env_clear, _temporary_env_override
from openpi.training import config as train_config
from openpi.policies import policy_config

checkpoint = Path("~/VLM_REPO/openpi/ckpts/pi05_libero_torch").expanduser()
cfg = train_config.get_config("pi05_libero")

print("模拟加载 teacher (duquant_enabled=False):")
with _temporary_env_clear("OPENPI_DUQUANT_"):
    print(f"  清除 DUQUANT 变量后:")
    remaining = {k: v for k, v in os.environ.items() if 'DUQUANT' in k}
    if remaining:
        print(f"    ⚠️  仍有 DUQUANT 变量: {list(remaining.keys())}")
    else:
        print(f"    ✅ DUQUANT 变量已清除")

    print(f"  加载 teacher model...")
    try:
        teacher_policy = policy_config.create_trained_policy(cfg, checkpoint, pytorch_device="cpu")
        teacher_model = teacher_policy._model

        # 检查是否有 DuQuant 层
        has_duquant = False
        for name, module in teacher_model.named_modules():
            if 'DuQuant' in module.__class__.__name__:
                has_duquant = True
                print(f"    ❌ Teacher 有 DuQuant 层: {name}")
                break

        if not has_duquant:
            print(f"    ✅ Teacher 没有 DuQuant 层（正确）")

    except Exception as e:
        print(f"    加载失败: {e}")

print()
print("模拟加载 quant (duquant_enabled=True):")
with _temporary_env_override("ATM_ENABLE", "0"):
    print(f"  临时设置 ATM_ENABLE=0")
    print(f"  保留其他环境变量")
    duquant_vars_now = {k: v for k, v in os.environ.items() if 'DUQUANT' in k}
    if duquant_vars_now:
        print(f"    当前 DUQUANT 变量: {list(duquant_vars_now.keys())}")
    else:
        print(f"    ❌ 没有 DUQUANT 变量！quant_model 不会被量化！")

    print(f"  加载 quant model...")
    try:
        quant_policy = policy_config.create_trained_policy(cfg, checkpoint, pytorch_device="cpu")
        quant_model = quant_policy._model

        # 检查是否有 DuQuant 层
        has_duquant = False
        duquant_layer_name = None
        for name, module in quant_model.named_modules():
            if 'DuQuant' in module.__class__.__name__:
                has_duquant = True
                duquant_layer_name = name
                break

        if has_duquant:
            print(f"    ✅ Quant 有 DuQuant 层: {duquant_layer_name} ({module.__class__.__name__})")
        else:
            print(f"    ❌ Quant 没有 DuQuant 层（错误！应该有量化）")
            print(f"    这会导致 teacher 和 quant 完全一样 → alpha 都是 1.0")

    except Exception as e:
        print(f"    加载失败: {e}")

print()
print("="*80)
print("结论:")
print("="*80)
if not duquant_vars:
    print("❌ 你没有设置 OPENPI_DUQUANT_* 环境变量")
    print("   这会导致校准时 quant_model 不被量化")
    print("   teacher 和 quant 模型会完全一样")
    print("   所有 alpha 都会是 1.0")
    print()
    print("解决方案:")
    print("  在运行 calibrate_atm_dit.py 之前，先设置环境变量:")
    print("    export OPENPI_DUQUANT_WBITS_DEFAULT=4")
    print("    export OPENPI_DUQUANT_ABITS=8")
    print("    export OPENPI_DUQUANT_BLOCK=64")
    print("    export OPENPI_DUQUANT_PERMUTE=1")
    print("    export OPENPI_DUQUANT_ROW_ROT=restore")
    print("    export OPENPI_DUQUANT_PACKDIR=/path/to/pack")
    print("    # ... 其他 DuQuant 变量")
    print("    python tools/calibrate_atm_dit.py ...")
else:
    print("✅ 你已经设置了 OPENPI_DUQUANT_* 环境变量")
    print("   quant_model 应该会被正确量化")
    print("   如果 alpha 仍然都是 1.0，说明：")
    print("   - 量化对 attention logits 的影响确实很小")
    print("   - 或者校准数据量不足")
print("="*80)
