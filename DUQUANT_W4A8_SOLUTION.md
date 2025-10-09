# DuQuant W4A8 实现方案总结

## 问题：为什么BitBLAS后端还是很慢（200x）？

### 根本原因

**BitBLAS的INT4 CUDA内核需要LLVM后端**，但当前配置存在以下问题：

1. TVM编译时禁用了LLVM（为了避免libtinfo.so.5错误）
2. BitBLAS在创建INT4 Matmul算子时报错：`target.build.llvm is not enabled`
3. 代码自动回退到PyTorch fallback实现（所以慢200x）

### 技术细节

BitBLAS的INT4量化需要两个步骤：
```
1. 权重压缩 (CPU侧，需要LLVM编译)
   - 将FP16权重量化到INT4
   - 打包成特殊格式给GPU使用

2. INT4矩阵乘法 (GPU侧，需要CUDA+TileLang)
   - 使用INT4 Tensor Cores
   - 比FP16快3-5x
```

当前问题：步骤1需要LLVM，但TVM没有启用LLVM支持。

## 解决方案对比

### 方案1：完整集成BitBLAS INT4内核 ✅ 推荐长期方案

**步骤：**

1. 安装LLVM 14（系统级）：
   ```bash
   sudo apt-get update
   sudo apt-get install -y llvm-14 llvm-14-dev
   ```

2. 重新编译TVM（启用LLVM）：
   ```bash
   cd third_party/BitBLAS/3rdparty/tvm/build
   # 修改config.cmake: set(USE_LLVM /usr/bin/llvm-config-14)
   cmake .. && make -j8
   ```

3. 实现权重格式转换（DuQuant → BitBLAS）：
   - 修改 `bitblas_utils.py`
   - 添加权重unpacking和repacking逻辑
   - 估计工作量：200-300行代码，2-3天测试

**优点：**
- 真正的3-5x加速
- INT4 Tensor Core加速

**缺点：**
- 需要root权限安装LLVM
- 需要额外开发和测试时间
- 权重格式转换有一定复杂度

### 方案2：使用torch.compile优化当前fallback 🚀 推荐短期方案

无需重新编译，直接优化现有PyTorch实现：

```python
# 在 bitblas_fallback.py 中添加torch.compile
@torch.compile(mode="max-autotune")
def linear_w4a8_compiled(x, packed_w, s_w, s_a, bias=None):
    # 现有的PyTorch实现
    ...
```

**优点：**
- 无需重新编译TVM
- 无需root权限
- 5分钟即可完成
- 预期加速：5-10x（虽然不如真INT4，但比现在好很多）

**缺点：**
- 仍然无法达到INT4的3-5x加速
- 首次运行需要编译时间（30秒-1分钟）

### 方案3：直接使用FP16基线 💡 最简单

对于LIBERO评估任务，FP16已经足够快：
- 延迟：0.67 ms/call
- 无需量化开销
- 最稳定的方案

**优点：**
- 无需任何修改
- 性能最好
- 数值最稳定

**缺点：**
- 显存占用更大（2x INT4）
- 无法验证DuQuant W4A8的效果

## 推荐方案：方案2（torch.compile优化）

这是**性价比最高**的方案：

### 实现代码

修改 `src/openpi/models_pytorch/quant_backends/bitblas_fallback.py`:

```python
import torch
import torch.nn.functional as F

# 使用torch.compile优化INT4 unpacking和matmul
@torch.compile(mode="max-autotune", fullgraph=True)
def _linear_w4a8_kernel(x, packed_w, s_w, s_a, bias, out_features, in_features):
    """Optimized W4A8 linear kernel with torch.compile."""
    # Unpack INT4
    packed_int16 = packed_w.to(torch.int16)
    low = (packed_int16 & 0xF).to(torch.int8)
    high = ((packed_int16 >> 4) & 0xF).to(torch.int8)

    # Convert to signed INT4 range [-8, 7]
    low = torch.where(low > 7, low - 16, low)
    high = torch.where(high > 7, high - 16, high)

    # Interleave to get [out, in]
    q_w = torch.empty(out_features, in_features, dtype=torch.int8, device=packed_w.device)
    q_w[:, 0::2] = low
    q_w[:, 1::2] = high
    q_w = q_w.to(torch.float16)

    # Quantize activations
    inv_scale = torch.clamp(s_a, min=1e-6).reciprocal()
    x_q = torch.clamp(torch.round(x * inv_scale), -128, 127)
    x_dq = x_q * s_a

    # Matmul with scaling
    out = F.linear(x_dq, q_w * s_w.unsqueeze(1), bias)
    return out

def linear_w4a8(x, packed_w, s_w, s_a, *, bias=None):
    """PyTorch fallback for W4A8 linear with torch.compile optimization."""
    out_features, half_cols = packed_w.shape
    in_features = half_cols * 2

    # Flatten input
    original_shape = x.shape
    x_flat = x.reshape(-1, in_features)

    # Call optimized kernel
    out = _linear_w4a8_kernel(
        x_flat, packed_w, s_w, s_a, bias,
        out_features, in_features
    )

    # Restore shape
    return out.reshape(*original_shape[:-1], out_features)
```

### 预期效果

- **首次运行**：30-60秒编译时间
- **后续运行**：5-10x加速（从137ms降到15-25ms）
- **vs FP16**：仍然慢20-40x，但比现在好很多

### 使用方法

```bash
# 无需重新编译任何东西
export OPENPI_DUQUANT_BACKEND=bitblas
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model."
./examples/libero/run_llm_w4a8.sh
```

## 总结

### 如果你想要

**最快的性能** → 使用方案3（FP16基线）
**快速验证DuQuant** → 使用方案2（torch.compile优化）
**真正的INT4加速** → 使用方案1（完整BitBLAS集成，需要2-3天）

### 我的建议

1. **现在**：使用方案2（torch.compile），5分钟完成
2. **LIBERO评估**：使用方案3（FP16），最稳定
3. **未来优化**：如果有时间和root权限，实现方案1

---

**当前状态：**
- ✅ BitBLAS编译完成（TVM + TileLang）
- ✅ DuQuant W4A8量化正确（数值精度OK）
- ❌ INT4内核未激活（缺少LLVM支持）
- ⚠️  性能200x慢（PyTorch fallback）

**下一步：**
实现方案2的torch.compile优化
