#!/bin/bash
# ATM 校准 - 简化版（所有环境变量都在脚本内部设置）
set -e

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate

# 清除所有可能干扰的环境变量
for var in $(env | grep -E '^(OPENPI_DUQUANT|ATM)_' | cut -d= -f1); do
    unset "$var"
done

# 禁用 CUDA 相关警告和堆栈跟踪
export TORCH_CUDA_GRAPH_DISABLE=1
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1
export PYTHONWARNINGS="ignore"

export PYTHONPATH=$PWD/src:$PWD/third_party/libero:$PWD

CKPT="${CKPT:-~/VLM_REPO/openpi/ckpts/pi05_libero_torch}"
STEPS="${STEPS:-32}"  # 默认 64 步（约 1 小时）
OUT="${OUT:-atm_alpha_dit_new.json}"

echo "="*80
echo "ATM 校准（简化版）"
echo "="*80
echo ""
echo "配置:"
echo "  Checkpoint: $CKPT"
echo "  Steps: $STEPS"
echo "  Output: $OUT"
echo ""
echo "⚠️  重要:"
echo "  - DuQuant 参数已内置到脚本中（CALIB_STEPS=8 加速）"
echo "  - 第一步会比较慢（约 5-10 分钟），之后会变快"
echo "  - 预计总时间: $STEPS 步约需 30-60 分钟"
echo ""
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
