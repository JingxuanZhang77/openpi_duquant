#!/usr/bin/env python3
"""Test DuQuant W4A8 (Fake) vs FP16 Baseline MSE.

使用与 run_quantvla.sh 相同的 DuQuant 配置:
- WBITS=4, ABITS=8
- BLOCK=64
- PERMUTE=1
- ROW_ROT=restore
- ACT_PCT=99.9
"""

import os
import sys

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F

DEVICE = "cuda"
PACK_DIR = "duquant_packed_full_llm_dit_mlp_w4a8_atm"

print("=" * 70)
print("DuQuant W4A8 (Fake) vs FP16 Baseline MSE")
print("=" * 70)

# ============================================================================
# 1. 加载 DuQuant pack 和模型
# ============================================================================
print("\n[1] Loading model and DuQuant pack...")

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.models_pytorch.duquant_preprocess import load_pack

config = _config.get_config("pi05_libero")
checkpoint_dir = os.path.expanduser("~/VLM_REPO/openpi/ckpts/pi05_libero_torch")

policy = _policy_config.create_trained_policy(config, checkpoint_dir)
model = policy._model

print(f"    Model loaded: {type(model).__name__}")

# 找到一个有 pack 的层
pack_files = [f for f in os.listdir(PACK_DIR) if f.endswith(".npz")]
if not pack_files:
    print(f"    ERROR: No pack files in {PACK_DIR}")
    sys.exit(1)

layer_name = pack_files[0].replace(".npz", "")
pack = load_pack(layer_name, PACK_DIR)
print(f"    Layer: {layer_name}")
print(f"    Pack: perm={pack.perm is not None}, R_in={len(pack.R_in_blocks) if pack.R_in_blocks else 0}, R_out={len(pack.R_out_blocks) if pack.R_out_blocks else 0}")

# ============================================================================
# 2. 获取真实权重
# ============================================================================
print("\n[2] Getting real weight...")

parts = layer_name.split(".")
layer = model
for part in parts:
    layer = getattr(layer, part)

W_fp16 = layer.weight.data.clone().to(DEVICE).half()
in_features = layer.in_features
out_features = layer.out_features

print(f"    W shape: ({out_features}, {in_features})")
print(f"    W range: [{W_fp16.min().item():.4f}, {W_fp16.max().item():.4f}]")

# ============================================================================
# 3. 生成测试激活
# ============================================================================
print("\n[3] Generating test activation...")

torch.manual_seed(42)
BATCH_SIZE = 16
x = torch.randn(BATCH_SIZE, in_features, device=DEVICE, dtype=torch.float16)
print(f"    x shape: ({BATCH_SIZE}, {in_features})")
print(f"    x range: [{x.min().item():.4f}, {x.max().item():.4f}]")

# ============================================================================
# 4. FP16 Baseline
# ============================================================================
print("\n[4] Computing FP16 Baseline...")

with torch.no_grad():
    y_baseline = F.linear(x, W_fp16)
print(f"    Baseline output range: [{y_baseline.min().item():.4f}, {y_baseline.max().item():.4f}]")

# ============================================================================
# 5. Fake DuQuant W4A8 (使用正确的 DuQuant 实现)
# ============================================================================
print("\n[5] Computing Fake DuQuant W4A8 (correct implementation)...")

from openpi.models_pytorch.duquant_preprocess import (
    apply_input_transform_optimized,
    apply_output_restore_optimized,
    transform_weight_for_forward_optimized,
)
from openpi.models_pytorch.duquant_layers import fake_quantize_sym

block_size = pack.meta.get("block_size", 64)
block_out_size = pack.meta.get("block_out_size", block_size)
wbits = pack.meta.get("wbits", 4)
abits = pack.meta.get("abits", 8)

print(f"    Config: wbits={wbits}, abits={abits}, block_size={block_size}")

# Step 1: 变换权重 (使用正确的函数)
W_t, w_scales = transform_weight_for_forward_optimized(
    W_fp16,
    pack,
    weight_bits=wbits,
    apply_row_rot=True,  # ROW_ROT=restore
    perm_cache=None,
    R_in_cache=None,
    R_out_cache=None,
    block_size=block_size,
    block_out_size=block_out_size,
)

# Step 2: 量化权重
W_t_quantized = fake_quantize_sym(W_t, w_scales[:, None], wbits, label="weight")

# Step 3: 变换激活
x_t = apply_input_transform_optimized(x, pack, perm_cache=None, R_in_cache=None, block_size=block_size)

# Step 4: 量化激活 (A8)
act_absmax = x_t.abs().max()
act_scale = act_absmax / 127.0
x_t_quantized = fake_quantize_sym(x_t, act_scale, abits, label="activation")

# Step 5: Matmul
y_t = F.linear(x_t_quantized, W_t_quantized)

# Step 6: 恢复输出 (ROW_ROT=restore)
y_duquant = apply_output_restore_optimized(y_t, pack, R_out_cache=None, block_out_size=block_out_size)

print(f"    DuQuant output range: [{y_duquant.min().item():.4f}, {y_duquant.max().item():.4f}]")

# ============================================================================
# 6. 比较
# ============================================================================
print("\n[6] Comparing results...")
print("-" * 70)

y_baseline_f = y_baseline.float()
y_duquant_f = y_duquant.float()

mse = ((y_duquant_f - y_baseline_f) ** 2).mean().item()
max_diff = (y_duquant_f - y_baseline_f).abs().max().item()

# Relative error
y_abs = y_baseline_f.abs()
mask = y_abs > 1e-6
if mask.sum() > 0:
    rel_err = ((y_duquant_f - y_baseline_f).abs()[mask] / y_abs[mask]).mean().item() * 100
else:
    rel_err = 0.0

print(f"DuQuant W4A8 vs FP16 Baseline:")
print(f"    MSE:      {mse:.6e}")
print(f"    Max diff: {max_diff:.6e}")
print(f"    Rel err:  {rel_err:.2f}%")

# ============================================================================
# 7. 对比无 DuQuant 变换的直接量化 (Naive W4A8)
# ============================================================================
print("\n[7] Comparing with naive W4A8 (no perm/rotation)...")

from openpi.models_pytorch.duquant_preprocess import compute_mse_scales

# Naive: 直接量化，不用 perm/rotation
max_q_w = (1 << (wbits - 1)) - 1  # 7 for 4-bit
weight_scale_naive = compute_mse_scales(W_fp16, wbits)
W_naive_quantized = fake_quantize_sym(W_fp16, weight_scale_naive[:, None], wbits, label="naive_weight")

max_q_a = 127
act_absmax_naive = x.abs().max()
act_scale_naive = act_absmax_naive / max_q_a
x_naive_quantized = fake_quantize_sym(x, act_scale_naive, abits, label="naive_act")

y_naive = F.linear(x_naive_quantized, W_naive_quantized)

mse_naive = ((y_naive.float() - y_baseline_f) ** 2).mean().item()
max_diff_naive = (y_naive.float() - y_baseline_f).abs().max().item()
if mask.sum() > 0:
    rel_err_naive = ((y_naive.float() - y_baseline_f).abs()[mask] / y_abs[mask]).mean().item() * 100
else:
    rel_err_naive = 0.0

print(f"Naive W4A8 (no perm/rotation) vs FP16 Baseline:")
print(f"    MSE:      {mse_naive:.6e}")
print(f"    Max diff: {max_diff_naive:.6e}")
print(f"    Rel err:  {rel_err_naive:.2f}%")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print(f"Layer: {layer_name}")
print(f"DuQuant config: wbits={wbits}, abits={abits}, block={block_size}")
print(f"perm={pack.perm is not None}, R_in={len(pack.R_in_blocks) if pack.R_in_blocks else 0}, R_out={len(pack.R_out_blocks) if pack.R_out_blocks else 0}")
print()
print(f"                        MSE          Rel Err")
print(f"DuQuant W4A8:          {mse:.2e}     {rel_err:.2f}%")
print(f"Naive W4A8:            {mse_naive:.2e}     {rel_err_naive:.2f}%")
if mse < mse_naive:
    print(f"Improvement:           {mse_naive/mse:.1f}x lower MSE with DuQuant")
else:
    print(f"WARNING: DuQuant worse than Naive!")
