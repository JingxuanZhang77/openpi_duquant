#!/bin/bash
# ATM 校准 - 用于完全量化版本（LLM + DiT 所有层）
# 使用已经打包好的 duquant_packed_full_llm_dit_w4a8_atm

set -e

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate

export PYTHONPATH=$PWD:$PWD/src:$PWD/third_party/libero

# 从 run_full_llm_dit_w4a8.sh 复制的完整 DuQuant 配置
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert."
export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=64
export OPENPI_DUQUANT_PERMUTE=1
export OPENPI_DUQUANT_ROW_ROT=restore
export OPENPI_DUQUANT_ACT_PCT=99.9
export OPENPI_DUQUANT_CALIB_STEPS=32
export OPENPI_DUQUANT_LS=0.15
export OPENPI_DUQUANT_PACKDIR=$PWD/duquant_packed_full_llm_dit_w4a8_atm
export OPENPI_DUQUANT_INCLUDE='(.*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*|.*gemma_expert.*(q_proj|k_proj|gate_proj|up_proj|down_proj).*)'
export OPENPI_DUQUANT_EXCLUDE='(?:^|\.)(norm|ln|layernorm|emb|embed|vision_tower|vision|multi_modal_projector|lm_head)(?:\.|$)'

# 禁用 CUDA warnings
export TORCH_CUDA_GRAPH_DISABLE=1
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1

STEPS="${STEPS:-32}"
OUT="${OUT:-atm_alpha_qk_mlp_dit.json}"

echo "="*80
echo "ATM 校准 - 完全量化版本"
echo "="*80
echo ""
echo "配置:"
echo "  PACKDIR: $OPENPI_DUQUANT_PACKDIR"
echo "  Steps: $STEPS"
echo "  Output: $OUT"
echo ""
echo "量化范围:"
echo "  - LLM: 所有 QKV + MLP（126 层）"
echo "  - DiT: 所有 QKV + MLP（126 层）"
echo "  - 总计: 252 层量化"
echo ""
echo "="*80
echo ""

time python tools/calibrate_atm_dit.py \
  --teacher-checkpoint ckpts/pi05_libero_torch \
  --quant-checkpoint   ckpts/pi05_libero_torch \
  --steps "$STEPS" \
  --out "$OUT" \
  --seed 42

echo ""
echo "="*80
echo "✅ 校准完成！"
echo "="*80
