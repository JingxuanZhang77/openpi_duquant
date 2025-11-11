#!/bin/bash
# ATM 校准 - 保证正确的版本
#
# 这个脚本会：
# 1. 设置所有必需的 DuQuant 环境变量
# 2. 调用 calibrate_atm_dit.py（原始版本）
# 3. 生成 atm_alpha_dit.json

set -e

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate

echo "="*80
echo "ATM 校准 - 设置环境变量"
echo "="*80

# 设置 DuQuant 环境变量（与运行脚本完全一致）
export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=64
export OPENPI_DUQUANT_PERMUTE=1
export OPENPI_DUQUANT_ROW_ROT=restore
export OPENPI_DUQUANT_ACT_PCT=99.9
export OPENPI_DUQUANT_CALIB_STEPS=32
export OPENPI_DUQUANT_LS=0.15
export OPENPI_DUQUANT_PACKDIR="$HOME/VLM_REPO/openpi/duquant_packed_atm_calib"
export OPENPI_DUQUANT_SCOPE=""
export OPENPI_DUQUANT_INCLUDE='.*(language_model\.(.*\.)?(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|gemma_expert\.model\.layers\.\d+\.(gate_proj|up_proj|down_proj)).*'
export OPENPI_DUQUANT_EXCLUDE='(?:^|\.)(vision_tower|vision|multi_modal_projector|norm|ln|layernorm|embed_tokens|lm_head)(?:\.|$)'

export PYTHONPATH=$PWD/src:$PWD/third_party/libero:$PWD

CKPT="${CKPT:-~/VLM_REPO/openpi/ckpts/pi05_libero_torch}"
STEPS="${STEPS:-256}"
OUT="${OUT:-atm_alpha_dit_final.json}"

echo ""
echo "配置:"
echo "  Checkpoint: $CKPT"
echo "  Steps: $STEPS"
echo "  Output: $OUT"
echo "  DuQuant: WBITS=$OPENPI_DUQUANT_WBITS_DEFAULT, ABITS=$OPENPI_DUQUANT_ABITS, BLOCK=$OPENPI_DUQUANT_BLOCK"
echo ""
echo "⚠️  重要:"
echo "  - Teacher 模型: FP16（脚本会清除 DuQuant 变量）"
echo "  - Quant 模型: W4A8（使用上面设置的变量）"
echo "  - 只计算 DiT attention 的 logits std"
echo ""
echo "预期时间: 约 30-60 分钟（256 步）"
echo "="*80
echo ""

# 运行校准
time python tools/calibrate_atm_dit.py \
  --teacher-checkpoint "$CKPT" \
  --quant-checkpoint "$CKPT" \
  --steps "$STEPS" \
  --out "$OUT" \
  --seed 42

echo ""
echo "="*80
echo "✅ 校准完成！"
echo "="*80
echo ""
echo "分析结果:"
python3 <<'PY'
import json
import numpy as np
import sys

out_file = sys.argv[1] if len(sys.argv) > 1 else "atm_alpha_dit_final.json"

try:
    with open(out_file) as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"❌ 文件不存在: {out_file}")
    sys.exit(1)

all_alphas = []
for key, value in data.items():
    if 'self_attn' in key and 'all' in value:
        all_alphas.extend(value['all'])

all_alphas = np.array(all_alphas)

print(f"总 heads: {len(all_alphas)}")
print(f"均值: {all_alphas.mean():.4f}")
print(f"标准差: {all_alphas.std():.4f}")
print(f"范围: [{all_alphas.min():.4f}, {all_alphas.max():.4f}]")
print()

num_one = np.sum(all_alphas == 1.0)
num_dev_5 = np.sum(np.abs(all_alphas - 1.0) > 0.05)
num_dev_10 = np.sum(np.abs(all_alphas - 1.0) > 0.10)

print(f"alpha = 1.0: {num_one}/{len(all_alphas)} ({num_one/len(all_alphas)*100:.1f}%)")
print(f"|alpha - 1.0| > 0.05: {num_dev_5}/{len(all_alphas)} ({num_dev_5/len(all_alphas)*100:.1f}%)")
print(f"|alpha - 1.0| > 0.10: {num_dev_10}/{len(all_alphas)}")
print()

if num_dev_5 > len(all_alphas) * 0.1:
    print("✅ 发现显著偏离！ATM 应该有效")
else:
    print("⚠️  大部分 alpha ≈ 1.0")
    print("   原因: DiT attention (QKV) 没有被量化")
    print("   只量化了: LLM (QKV+MLP) + DiT (MLP)")
PY "$OUT"
