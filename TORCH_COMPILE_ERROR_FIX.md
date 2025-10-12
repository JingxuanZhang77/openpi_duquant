# Torch.Compile Error Fix: _act_scale Cache Issue

## 🐛 错误信息

```
File "/home/jz97/VLM_REPO/openpi/src/openpi/models_pytorch/duquant_layers.py", line 252, in forward
    s_a = self._get_act_scale(x_t)
  File "/home/jz97/VLM_REPO/openpi/src/openpi/models_pytorch/duquant_layers.py", line 238, in _get_act_scale
    self._act_scale = scale.to(dtype=x.dtype, device=x.device)

To prevent overwriting, clone the tensor outside of torch.compile()
or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.
```

## 🔍 原因分析

### 问题本质

**Torch.compile不允许在编译的函数内直接修改模块的属性（mutation）**

在 `_get_act_scale()` 方法中：
```python
# Line 228 & 238
self._act_scale = scale.to(dtype=x.dtype, device=x.device)
```

这行代码在 `forward()` 函数被 `torch.compile()` 编译后，尝试修改 `self._act_scale`，违反了torch.compile的限制。

### Torch.Compile的限制

1. **不允许mutation**：编译后的函数不能修改模块的state
2. **需要functional**：所有操作都应该是pure function
3. **缓存会破坏graph**：动态修改属性会导致recompilation

### 为什么之前没报错？

之前禁用了torch.compile：
```bash
export OPENPI_DISABLE_TORCH_COMPILE=1
```

启用torch.compile后，这个问题就暴露了。

---

## 🔧 解决方案

有3种方案，按推荐顺序：

### 方案1: 使用register_buffer + in-place操作 ⭐⭐⭐⭐⭐

**最推荐：torch.compile友好，性能最佳**

修改 `duquant_layers.py`:

```python
def __init__(self, base: nn.Linear, name: str, cfg: DuQuantConfig, ...):
    # ... 原有代码 ...

    # 改用register_buffer存储act_scale（而不是普通属性）
    self.register_buffer("_act_scale", None)
    self._act_scale_initialized = False  # 用flag而不是检查None

def _get_act_scale(self, x: torch.Tensor) -> torch.Tensor:
    if self.cfg.act_bits <= 0:
        return torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)

    # 如果已初始化，直接返回
    if self._act_scale_initialized:
        return self._act_scale

    # 初始化时使用no_grad和in-place操作
    with torch.no_grad():
        if self.calibrator is not None and not self.calibrator.is_full():
            self.calibrator.observe(x)
            if self.calibrator.is_full():
                p_vec = self.calibrator.finalize()
                max_q = qmax(self.cfg.act_bits)
                scale = torch.clamp(p_vec / max_q, min=1e-6)
                # 使用copy_替代直接赋值
                if self._act_scale is None:
                    self._act_scale = scale.to(dtype=x.dtype, device=x.device)
                else:
                    self._act_scale.copy_(scale.to(dtype=x.dtype, device=x.device))
                self._act_scale_initialized = True

        # Fallback
        if not self._act_scale_initialized:
            x_abs = torch.abs(x.detach().to(torch.float32))
            C = x_abs.shape[-1]
            x2d = x_abs.reshape(-1, C)
            p_vec = torch.quantile(x2d, self.cfg.act_percentile / 100.0, dim=0)
            max_q = qmax(self.cfg.act_bits)
            scale = torch.clamp(p_vec / max_q, min=1e-6)
            if self._act_scale is None:
                self._act_scale = scale.to(dtype=x.dtype, device=x.device)
            else:
                self._act_scale.copy_(scale.to(dtype=x.dtype, device=x.device))
            self._act_scale_initialized = True

    return self._act_scale
```

**优点：**
- ✅ Torch.compile友好
- ✅ 性能最佳
- ✅ 保持原有逻辑

---

### 方案2: 禁用activation quantization ⭐⭐⭐⭐

**最简单：暂时禁用激活量化，只保留权重量化**

```bash
# 在run_optimized_duquant.sh中添加：
export OPENPI_DUQUANT_ABITS=16  # 禁用激活量化（16bit = no quant）
```

**优点：**
- ✅ 立即生效，无需修改代码
- ✅ 仍然测试权重量化（W4）
- ⚠️ 无法测试完整W4A8

**影响：**
- 只测试W4，不测试A8
- 精度可能略好（激活未量化）
- 仍能获得torch.compile加速

---

### 方案3: 预先初始化act_scale ⭐⭐⭐

**Warmup：在编译前初始化所有缓存**

修改 `enable_duquant_if_configured()`:

```python
def enable_duquant_if_configured(model: nn.Module) -> None:
    # ... 原有代码 ...

    wrap_duquant(model, layer_names, cfg, per_layer_wbits, dry_run=False)

    # NEW: Warmup所有DuQuant层
    print("[DUQUANT] Warming up activation scales...")
    with torch.no_grad():
        # 创建dummy input
        dummy_input = torch.randn(1, 1024, device='cuda', dtype=torch.bfloat16)
        for name, module in model.named_modules():
            if isinstance(module, DuQuantLinear):
                # 触发_get_act_scale初始化
                _ = module._get_act_scale(dummy_input[:, :module.in_features])
    print("[DUQUANT] Warmup complete!")
```

**优点：**
- ✅ 在torch.compile之前完成所有初始化
- ✅ 保持完整W4A8

**缺点：**
- ⚠️ 需要修改代码
- ⚠️ Warmup可能不准确（dummy data）

---

### 方案4: 禁用torch.compile ⭐

**回退：如果修复太复杂**

```bash
# 保持原样
export OPENPI_DISABLE_TORCH_COMPILE=1
```

**优点：**
- ✅ 立即生效
- ✅ 无需修改任何代码

**缺点：**
- ❌ 失去20-40x加速
- ❌ 回到原来的慢速度

---

## 🎯 推荐修复顺序

### 快速测试（5分钟）

**先用方案2：禁用激活量化**

```bash
# 编辑 run_optimized_duquant.sh
export OPENPI_DUQUANT_ABITS=16  # 添加这行

# 运行测试
bash examples/libero/run_optimized_duquant.sh
```

**结果：**
- 如果能正常运行 → 问题确认，只是act_scale的问题
- 仍能获得torch.compile加速（测试W4）

---

### 完整修复（30分钟）

**实现方案1：修改duquant_layers.py**

我会提供完整的patch文件。

---

## 📝 临时Workaround

如果你现在就想运行，最简单的办法：

```bash
# 方法A: 禁用激活量化（推荐）
export OPENPI_DUQUANT_ABITS=16
bash examples/libero/run_optimized_duquant.sh

# 方法B: 禁用torch.compile（不推荐，失去加速）
export OPENPI_DISABLE_TORCH_COMPILE=1
bash examples/libero/run_optimized_duquant.sh
```

---

## 🔬 为什么register_buffer有效？

```python
# 普通属性（会报错）
self._act_scale = tensor  # ❌ Mutation，torch.compile不允许

# register_buffer（torch.compile友好）
self.register_buffer("_act_scale", tensor)  # ✅ 被识别为模块状态
self._act_scale.copy_(tensor)  # ✅ In-place更新，不改变引用
```

**关键差异：**
- `self.attr = tensor` → 改变Python对象的引用（mutation）
- `self.attr.copy_(tensor)` → in-place更新tensor内容（allowed）

Torch.compile允许in-place操作，但不允许改变对象引用。

---

## 🚀 我来帮你修复

我现在就可以帮你修改代码，选择：

1. **快速测试**：我帮你添加 `OPENPI_DUQUANT_ABITS=16` 到脚本
2. **完整修复**：我修改 `duquant_layers.py` 实现方案1

你想要哪个？
