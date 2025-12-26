#!/usr/bin/env python3
"""单层 W8A8 比较测试 - BitBLAS vs Fake W8A8

使用固定随机种子生成 W 和 x，比较 BitBLAS INT8 kernel 和 Fake W8A8 的 MSE。
"""

import torch
import torch.nn.functional as F

DEVICE = "cuda"

# 固定随机种子
torch.manual_seed(42)

# 模拟真实层的维度 (MLP gate_proj: 2048 -> 16384)
BATCH_SIZE = 16
IN_FEATURES = 2048
OUT_FEATURES = 8192

print("=" * 70)
print("Single Layer W8A8 Comparison: BitBLAS vs Fake W8A8")
print(f"Dimensions: ({BATCH_SIZE}, {IN_FEATURES}) x ({OUT_FEATURES}, {IN_FEATURES})")
print("=" * 70)

# ============================================================================
# 1. 生成 W 和 x
# ============================================================================
print("\n[1] Generating W and x...")

# 权重: 模拟真实权重分布 (均值0, 标准差~0.01)
W = torch.randn(OUT_FEATURES, IN_FEATURES, device=DEVICE, dtype=torch.float32) * 0.01

# 激活: 模拟真实激活分布
x = torch.randn(BATCH_SIZE, IN_FEATURES, device=DEVICE, dtype=torch.float32)

print(f"    W: shape={W.shape}, range=[{W.min():.4f}, {W.max():.4f}]")
print(f"    x: shape={x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")

# ============================================================================
# 2. 量化参数
# ============================================================================
print("\n[2] Computing quantization parameters...")

# 权重量化: per-channel absmax
weight_absmax = W.abs().max(dim=1)[0]
weight_scale = (weight_absmax / 127.0).clamp(min=1e-8)

# 激活量化: per-tensor absmax
act_absmax = x.abs().max()
act_scale = (act_absmax / 127.0).clamp(min=1e-8)

# 量化
W_q = (W / weight_scale[:, None]).round().clamp(-127, 127)
x_q = (x / act_scale).round().clamp(-127, 127)

print(f"    weight_scale: range=[{weight_scale.min():.6f}, {weight_scale.max():.6f}]")
print(f"    act_scale: {act_scale.item():.6f}")

# ============================================================================
# 3. Fake W8A8 (FP32 模拟)
# ============================================================================
print("\n[3] Computing Fake W8A8...")

y_fake = F.linear(x_q, W_q)
y_fake = y_fake * (act_scale * weight_scale)

print(f"    Fake W8A8 output: range=[{y_fake.min():.4f}, {y_fake.max():.4f}]")

# ============================================================================
# 4. BitBLAS W8A8
# ============================================================================
print("\n[4] Computing BitBLAS W8A8...")

try:
    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target

    target = auto_detect_nvidia_target()
    print(f"    BitBLAS target: {target}")

    matmul_config = MatmulConfig(
        M=[BATCH_SIZE],
        N=OUT_FEATURES,
        K=IN_FEATURES,
        A_dtype="int8",
        W_dtype="int8",
        out_dtype="int32",
        accum_dtype="int32",
        layout="nt",
        with_scaling=False,
        with_bias=False,
    )

    bitblas_matmul = Matmul(matmul_config, target=target, backend="tir", enable_tuning=False)

    # 准备 INT8 数据
    W_int8 = W_q.to(torch.int8)
    x_int8 = x_q.to(torch.int8)

    # Transform weight
    qweight = bitblas_matmul.transform_weight(W_int8)
    if isinstance(qweight, list):
        qweight = qweight[0]

    # 执行 INT8 matmul
    output_int32 = torch.empty(BATCH_SIZE, OUT_FEATURES, dtype=torch.int32, device=DEVICE)
    bitblas_matmul(x_int8, qweight, output=output_int32)

    # 反量化
    y_bitblas = output_int32.float() * (act_scale * weight_scale)

    print(f"    BitBLAS output: range=[{y_bitblas.min():.4f}, {y_bitblas.max():.4f}]")
    BITBLAS_OK = True

except Exception as e:
    print(f"    BitBLAS error: {e}")
    BITBLAS_OK = False

# ============================================================================
# 5. 比较
# ============================================================================
if BITBLAS_OK:
    print("\n[5] Comparing results...")
    print("-" * 70)

    mse = ((y_bitblas - y_fake) ** 2).mean().item()
    max_diff = (y_bitblas - y_fake).abs().max().item()

    print(f"BitBLAS vs Fake W8A8:")
    print(f"    MSE:      {mse:.6e}")
    print(f"    Max diff: {max_diff:.6e}")

    print("\n" + "=" * 70)
    if mse < 1e-10:
        print("✅ SUCCESS: MSE < 1e-10, BitBLAS matches Fake W8A8 exactly!")
    elif mse < 1e-5:
        print("✅ SUCCESS: MSE < 1e-5, BitBLAS matches Fake W8A8 closely!")
    else:
        print(f"⚠ WARNING: MSE = {mse:.2e}")
    print("=" * 70)
