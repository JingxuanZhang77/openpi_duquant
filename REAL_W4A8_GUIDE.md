# 如何实现真正的DuQuant W4A8加速

## 问题总结

BitBLAS的INT4内核需要：
1. **完整的LLVM工具链**（llvm-config, llvm-14-dev）
2. **CUDA + cuBLAS + TileLang**
3. **权重格式转换**（DuQuant → BitBLAS）

当前问题：Conda环境缺少llvm-config，导致TVM无法启用LLVM后端。

## 三种实现方案

### 方案1：使用系统LLVM（需要root权限）⭐ 最彻底

```bash
# 1. 安装系统LLVM
sudo apt-get update
sudo apt-get install -y llvm-14 llvm-14-dev clang-14

# 2. 运行安装脚本
./install_bitblas_real_w4a8.sh

# 3. 测试
python test_bitblas_int4.py
```

**优点**：真正的INT4 Tensor Core加速（3-5x vs FP16）
**缺点**：需要root权限

### 方案2：使用torch.compile优化（无需root）⭐⭐ 推荐

无需BitBLAS，直接优化PyTorch实现：

```python
# 修改 src/openpi/models_pytorch/quant_backends/bitblas_fallback.py

import torch
import torch.nn.functional as F

@torch.compile(mode="max-autotune", fullgraph=True)
def _w4a8_matmul_kernel(x_flat, packed_w, s_w, s_a, out_features, in_features):
    """torch.compile优化的W4A8内核"""
    # Unpack INT4
    low = (packed_w & 0xF).to(torch.int8)
    high = ((packed_w >> 4) & 0xF).to(torch.int8)
    low = torch.where(low > 7, low - 16, low)
    high = torch.where(high > 7, high - 16, high)

    # Reconstruct weights [out, in]
    q_w = torch.empty(out_features, in_features, dtype=torch.float16, device=packed_w.device)
    q_w[:, 0::2] = low.to(torch.float16)
    q_w[:, 1::2] = high.to(torch.float16)

    # Quantize activations
    inv_scale = torch.clamp(s_a, min=1e-6).reciprocal()
    x_q = torch.clamp(torch.round(x_flat * inv_scale), -128, 127)
    x_dq = (x_q * s_a).to(torch.float16)

    # Fused matmul
    return F.linear(x_dq, q_w * s_w.unsqueeze(1))

def linear_w4a8(x, packed_w, s_w, s_a, *, bias=None):
    out_features, half_cols = packed_w.shape
    in_features = half_cols * 2
    x_flat = x.reshape(-1, in_features)

    out = _w4a8_matmul_kernel(x_flat, packed_w, s_w, s_a, out_features, in_features)

    if bias is not None:
        out = out + bias

    return out.reshape(*x.shape[:-1], out_features)
```

**使用方法**：
```bash
# 无需任何编译，直接使用
export OPENPI_DUQUANT_BACKEND=bitblas
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model."
./examples/libero/run_llm_w4a8.sh
```

**预期性能**：
- 首次运行：30-60秒编译（自动缓存）
- 后续运行：5-10x加速（从137ms降到15-30ms）
- vs FP16：仍然慢20-40x，但比现在好很多

**优点**：
- ✅ 无需root权限
- ✅ 无需重新编译
- ✅ 5分钟实现
- ✅ 自动GPU kernel fusion

**缺点**：
- ❌ 仍然无法达到真INT4的3-5x加速

### 方案3：直接使用FP16（最简单）⭐⭐⭐

```bash
# 不使用DuQuant，直接用FP16
export OPENPI_DUQUANT_BACKEND=off
./examples/libero/run_llm_w4a8.sh
```

**性能**：0.67 ms/call（最快）
**显存**：比INT4多2x
**稳定性**：最高

## 我的最终建议

### 如果你有root权限
→ 使用**方案1**（系统LLVM + BitBLAS），获得真正的INT4加速

### 如果没有root权限
→ 使用**方案2**（torch.compile），5-10x加速已经够用

### 如果只是想跑LIBERO评估
→ 使用**方案3**（FP16），性能最好最稳定

## 快速实现方案2（推荐）

我现在帮你实现torch.compile优化版本？只需要：

1. 修改 `bitblas_fallback.py` (5分钟)
2. 测试性能 (2分钟)
3. 完成！

**预期加速**：从137ms降到15-30ms（5-10x加速）

想要我现在实现吗？
