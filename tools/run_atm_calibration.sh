#!/bin/bash
# 运行 ATM 校准 - 确保一个量化一个不量化
#
# 这个脚本会：
# 1. 不设置任何环境变量（让 calibrate_atm_dit.py 内部设置）
# 2. 运行校准，生成新的 atm_alpha_dit.json
# 3. 显示详细的 teacher vs quant 对比

set -e

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate

# 清除所有 DuQuant 和 ATM 环境变量（让脚本内部设置）
unset OPENPI_DUQUANT_WBITS_DEFAULT
unset OPENPI_DUQUANT_ABITS
unset OPENPI_DUQUANT_BLOCK
unset OPENPI_DUQUANT_PERMUTE
unset OPENPI_DUQUANT_ROW_ROT
unset OPENPI_DUQUANT_ACT_PCT
unset OPENPI_DUQUANT_CALIB_STEPS
unset OPENPI_DUQUANT_LS
unset OPENPI_DUQUANT_PACKDIR
unset OPENPI_DUQUANT_SCOPE
unset OPENPI_DUQUANT_INCLUDE
unset OPENPI_DUQUANT_EXCLUDE
unset OPENPI_DUQUANT_DEBUG
unset ATM_ENABLE
unset ATM_SCOPE
unset ATM_ALPHA_PATH

export PYTHONPATH=$PWD/src:$PWD/third_party/libero:$PWD

CKPT="${CKPT:-~/VLM_REPO/openpi/ckpts/pi05_libero_torch}"
STEPS="${STEPS:-32}"
OUT="${OUT:-atm_alpha_dit_fixed.json}"

echo "========================================"
echo "ATM Calibration (Fixed Version)"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Steps: $STEPS"
echo "Output: $OUT"
echo ""
echo "📝 注意："
echo "  - Teacher 模型：FP16/BF16（不量化）"
echo "  - Quant 模型：DuQuant W4A8（脚本内部自动设置）"
echo "  - 使用 block_size=64（与运行脚本一致）"
echo ""
echo "预期结果："
echo "  如果之前 alpha 都是 1.0 → 现在应该有显著偏离"
echo "  如果仍然接近 1.0 → 说明量化对 attention 影响确实很小"
echo "========================================"
echo ""

# 运行校准
time python tools/calibrate_atm_dit.py \
  --teacher-checkpoint "$CKPT" \
  --quant-checkpoint "$CKPT" \
  --steps "$STEPS" \
  --out "$OUT" \
  --seed 42

echo ""
echo "========================================"
echo "校准完成！"
echo "========================================"
echo ""

# 分析生成的 alpha
python3 <<'PY'
import json
import numpy as np
import sys

out_file = sys.argv[1] if len(sys.argv) > 1 else "atm_alpha_dit_fixed.json"

try:
    with open(out_file) as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"❌ 文件不存在: {out_file}")
    sys.exit(1)

print("="*80)
print(f"分析 {out_file}")
print("="*80)
print()

# 统计所有 self_attn 层的 alpha
all_alphas = []
layers_with_deviation = []

for key, value in data.items():
    if 'self_attn' in key and 'all' in value:
        alphas = np.array(value['all'])
        all_alphas.extend(alphas)

        if np.any(np.abs(alphas - 1.0) > 0.05):
            layers_with_deviation.append((key, alphas))

all_alphas = np.array(all_alphas)

print("Overall Statistics:")
print(f"  Total heads: {len(all_alphas)}")
print(f"  Mean: {all_alphas.mean():.4f}")
print(f"  Std: {all_alphas.std():.4f}")
print(f"  Min: {all_alphas.min():.4f}")
print(f"  Max: {all_alphas.max():.4f}")
print()

num_deviated = np.sum(np.abs(all_alphas - 1.0) > 0.05)
num_exact_one = np.sum(all_alphas == 1.0)

print(f"  Heads with alpha = 1.0: {num_exact_one} ({num_exact_one/len(all_alphas)*100:.1f}%)")
print(f"  Heads with |alpha - 1.0| > 0.05: {num_deviated} ({num_deviated/len(all_alphas)*100:.1f}%)")
print(f"  Heads with |alpha - 1.0| > 0.10: {np.sum(np.abs(all_alphas - 1.0) > 0.10)}")
print()

if num_deviated > 0:
    print("✅ 发现显著偏离！ATM 应该会有效果")
    print()
    print(f"有偏离的层（共 {len(layers_with_deviation)} 层）:")
    for layer_name, alphas in layers_with_deviation[:5]:  # 显示前 5 层
        print(f"  {layer_name}:")
        print(f"    alpha = {alphas}")
        print(f"    range = [{alphas.min():.4f}, {alphas.max():.4f}]")
else:
    print("⚠️  所有 alpha 都接近 1.0")
    print()
    print("可能原因:")
    print("  1. DuQuant 配置：只量化了 LLM + DiT MLP，DiT 注意力层未量化")
    print("  2. 因此注意力 logits 的温度偏移很小")
    print()
    print("ATM 效果预测:")
    print("  - 预计提升幅度有限（+0.5% 到 +2%）")
    print("  - 建议优先优化 DuQuant 配置（block_size、calibration）")

print()
print("="*80)
print("下一步:")
print("="*80)
print(f"1. 使用新的 alpha 运行评测:")
print(f"     export ATM_ENABLE=1")
print(f"     export ATM_ALPHA_PATH={out_file}")
print(f"     bash examples/libero/run_llm_dit_mlp_w4a8_atm.sh")
print()
print(f"2. 对比结果:")
print(f"     不用 ATM: bash examples/libero/run_llm_dit_mlp_w4a8.sh")
print(f"     用 ATM:   上面的命令")
print("="*80)
PY "$OUT"

echo ""
