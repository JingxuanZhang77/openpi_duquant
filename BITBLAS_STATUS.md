# BitBLAS W4A8 部署状态报告

## 问题诊断

### 原始错误
```
AttributeError: 'Matmul' object has no attribute 'torch_func'
```

### 根本原因

1. **编译失败** - BitBLAS 在 SM86 GPU (A40) 上尝试编译 TileLang backend 时遇到 CUTLASS/TMA 错误：
   ```
   identifier "CU_TENSOR_MAP_SWIZZLE_128B" is undefined
   identifier "CUtensorMap" is undefined
   ```
   - 这是因为 CUTLASS 生成了需要 CUDA 12.0+ 和 SM90 (H100) 的 TMA (Tensor Memory Accelerator) 代码
   - 即使强制 `target="cuda -arch=sm_80"` 也无法避免

2. **TIR backend bug** - 作为替代，使用 `backend="tir"` 可以编译成功，但 INT8xINT4 matmul 结果**完全错误**：
   - 所有输出列都是相同的值
   - 例如：期望 `[[10, 20, 30], [26, 52, 78]]`，实际输出 `[[10, 10, 10], [26, 26, 26]]`

### 已修复的问题

✅ **bitblas_backend.py 配置更新：**

1. 使用 `backend="tir"` 避免 CUTLASS 编译错误
2. 使用 `with_scaling=False` 并手动应用 scales（INT source format 不支持 `with_scaling=True`）
3. 设置 `target="cuda -arch=sm_80"` 作为默认值

但由于 TIR backend 的 bug，**BitBLAS W4A8 当前无法正常工作**。

## 测试文件

### 1. `test_duquant_bitblas.py`
- ✅ 编译成功（使用 TIR backend）
- ❌ 数值结果错误（TIR backend bug）

### 2. `testbitblas.py`
- 原始测试文件，包含两个测试
- 第一个测试（INT8xINT4）失败
- 第二个测试（FP16xINT4）成功

## 推荐的解决方案

### 方案 1：使用 Fake Quantization（推荐）

**优点：**
- ✅ 立即可用，无需等待 BitBLAS 修复
- ✅ 保留 DuQuant 的量化感知优化（permutation, rotation）
- ✅ 仍然能获得精度提升
- ✅ 稳定可靠

**缺点：**
- ❌ 无法获得真实 INT4 加速
- ❌ 内存占用与 FP16 相同

**使用方法：**
```bash
# 运行新创建的脚本
./scripts/run_llm_w4a8_fake.sh

# 或者手动设置
export OPENPI_DUQUANT_BACKEND=fake
bash examples/libero/run_llm_w4a8.sh
```

### 方案 2：等待 BitBLAS 修复

**需要做的：**
1. 向 BitBLAS 团队报告 TIR backend 的 INT8xINT4 bug
2. 或者等待他们修复 CUTLASS/TMA 在旧 GPU 上的兼容性问题

**时间线：** 未知

### 方案 3：使用其他 INT4 kernel

**选项：**
- GPTQ/Marlin kernels
- AutoGPTQ
- llm.c 的 INT4 kernels

**需要做的：**
- 实现新的 backend adapter
- 测试兼容性

## 修改的文件总结

### [bitblas_backend.py](src/openpi/models_pytorch/quant_backends/bitblas_backend.py)

**主要修改：**
```python
# 1. 使用 TIR backend 避免 CUTLASS 编译错误
cfg = bitblas.MatmulConfig(
    ...
    A_dtype="int8",
    W_dtype="int4",
    with_scaling=False,  # INT format 不支持内部 scaling
    ...
)

self._matmul = bitblas.Matmul(
    config=cfg,
    target="cuda -arch=sm_80",  # 默认 SM80 target
    backend="tir"               # 使用 TIR backend
)

# 2. 手动应用 scales
def run(self, x_t, s_a):
    Qa = quantize_to_int8(x_t, s_a)
    y = self._matmul(Qa, self._packed_w)  # 不传 scale 参数
    y = y * self._packed_scale.t()        # 手动应用 scales
    return y
```

**状态：** ⚠️ 已修复编译问题，但 TIR backend 有 bug

## 测试命令

```bash
# 1. 测试 BitBLAS 基础功能（会显示 TIR backend bug）
source examples/libero/.venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python test_duquant_bitblas.py

# 2. 运行实际的 LLM W4A8 with fake quantization
./scripts/run_llm_w4a8_fake.sh
```

## 环境信息

- GPU: A40 (SM86)
- CUDA: PyTorch 2.7.1+cu126
- BitBLAS: 从 third_party/BitBLAS 安装
- Python: 3.11 (.venv)

## 结论

**当前状态：** ❌ BitBLAS W4A8 在此环境中不可用

**建议：** ✅ 使用 fake quantization 方案继续测试 DuQuant LLM quantization

fake quantization 仍然能提供：
- ✅ Quantization-aware optimization (permutation, rotation)
- ✅ 更好的量化精度
- ✅ 为未来真实 INT4 部署做准备

只是暂时无法获得：
- ❌ 内存节省（仍然是 FP16 存储）
- ❌ INT4 计算加速

等 BitBLAS 修复后，可以无缝切换到真实 INT4 kernel。
