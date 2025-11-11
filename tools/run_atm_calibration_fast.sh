#!/bin/bash
# 快速测试版本 - 只用 16 步
set -e

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate

# 清除所有环境变量
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
STEPS=32  # 快速测试只用 16 步
SCOPE="${SCOPE:-dit}"  # 默认 dit，可以设置为 llm 或 all
OUT="atm_alpha_${SCOPE}_test.json"

echo "========================================"
echo "ATM Calibration - Fast Test (16 steps)"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Scope: $SCOPE (可用 SCOPE=llm 或 SCOPE=all 切换)"
echo "Output: $OUT"
echo ""
echo "这是快速测试版本，只用 16 步来验证是否能正常运行"
echo "如果成功，再用完整的 64 或 256 步"
echo "========================================"
echo ""

# 运行校准
time python tools/calibrate_atm_dit.py \
  --teacher-checkpoint "$CKPT" \
  --quant-checkpoint "$CKPT" \
  --steps "$STEPS" \
  --out "$OUT" \
  --scope "$SCOPE" \
  --seed 42

echo ""
echo "✅ 快速测试完成！"
echo ""
echo "如果运行正常，使用完整版本："
echo "  STEPS=64 bash tools/run_atm_calibration.sh"
