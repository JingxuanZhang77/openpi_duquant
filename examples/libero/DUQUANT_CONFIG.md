# DuQuant Configuration for LIBERO Headless Evaluation

## 两种 DuQuant 配置方式

headless 模式支持两种 DuQuant 配置：

### 配置 A：使用 DuQuant 内部默认值

```bash
OPENPI_DUQUANT_DEBUG=1
OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model."
# 不显式指定其他参数，使用 DuQuant 内部默认值：
# - BLOCK=16
# - PERMUTE=1 (启用)
# - ROW_ROT=restore
```

**特点**：
- 遵循 DuQuant 论文的原始设置
- 可能有更高的精度
- 推理可能稍慢（因为 permute=1 和 row-rot）

### 配置 B：显式自定义参数（推荐）

```bash
OPENPI_DUQUANT_DEBUG=1
OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model."
OPENPI_DUQUANT_WBITS_DEFAULT=4              # 权重4位
OPENPI_DUQUANT_ABITS=8                      # 激活8位
OPENPI_DUQUANT_ACT_PCT=98.0                 # 激活百分位
OPENPI_DUQUANT_BLOCK=64                     # 更大的 block size
OPENPI_DUQUANT_PERMUTE=0                    # 禁用排列
OPENPI_DUQUANT_ROW_ROT=0                    # 禁用 row rotation
OPENPI_DUQUANT_CALIB_STEPS=64               # 校准步数
```

**特点**：
- 更快的推理速度（permute=0, row-rot=0）
- 更大的 block size (64) 可能更适合现代硬件
- 显式控制所有参数，便于调优

### 配置对比表

| 参数 | 配置 A (默认) | 配置 B (自定义) | 说明 |
|------|--------------|----------------|------|
| `WBITS_DEFAULT` | (internal) | 4 | 配置A由DuQuant内部决定 |
| `ABITS` | (internal) | 8 | 配置A由DuQuant内部决定 |
| `BLOCK` | 16 | 64 | 更大block可能更快 |
| `PERMUTE` | 1 | 0 | 禁用排列可加速推理 |
| `ROW_ROT` | restore | 0 | 禁用rotation可加速推理 |
| `ACT_PCT` | - | 98.0 | 显式设置激活范围 |
| `CALIB_STEPS` | - | 64 | 显式设置校准步数 |
| **推理速度** | 中等 | 快 | |
| **精度** | 高 | 中高 | |
| **内存占用** | 低 | 低 | 两者相似 |

## 如何选择？

- **如果你要对比原始 DuQuant 论文结果** → 使用配置 A
- **如果你要最快的推理速度** → 使用配置 B
- **如果不确定** → 使用配置 B（默认）

## 参数说明

### 1. `OPENPI_DUQUANT_DEBUG=1`
- **作用**: 启用详细的调试日志
- **输出**: 显示哪些层被量化、量化参数、性能统计等
- **建议**: 首次运行时启用，确认量化正确应用

### 2. `OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model."`
- **作用**: 指定要量化的模块前缀
- **说明**: 只量化 PaliGemma 的 expert 模型部分
- **示例**:
  - `paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj` ✅ 会被量化
  - `paligemma_with_expert.vision_encoder.encoder.layers.0` ❌ 不会被量化

### 3. `OPENPI_DUQUANT_WBITS_DEFAULT=4`
- **作用**: 权重（weight）的量化位数
- **选项**: 通常为 2, 3, 4, 6, 8
- **说明**:
  - 4位 = 16个量化级别
  - 降低权重精度以节省内存
  - W4 通常能保持较好的性能

### 4. `OPENPI_DUQUANT_ABITS=8`
- **作用**: 激活值（activation）的量化位数
- **选项**: 通常为 8, 16（或不量化）
- **说明**:
  - 8位 = 256个量化级别
  - 激活值量化对精度影响较大，因此用8位
  - A8 在速度和精度间有较好平衡

### 5. `OPENPI_DUQUANT_ACT_PCT=98.0`
- **作用**: 激活值的百分位数，用于确定量化范围
- **说明**:
  - 使用第98百分位数作为激活值的最大值
  - 可以避免极端异常值影响量化精度
  - 98.0表示忽略最高2%的异常值

### 6. `OPENPI_DUQUANT_BLOCK=64`
- **作用**: 量化时的块大小（block size）
- **说明**:
  - 将权重矩阵分成 64×64 的块进行量化
  - 更小的块 = 更精细的量化 = 更高精度（但更慢）
  - 64 是常用的平衡选择

### 7. `OPENPI_DUQUANT_PERMUTE=0`
- **作用**: 是否对权重进行排列（permutation）
- **选项**:
  - `0` = 禁用排列
  - `1` = 启用排列
- **说明**:
  - 排列可以改善量化质量
  - 但会增加推理开销
  - 设置为0以获得更快的推理速度

### 8. `OPENPI_DUQUANT_ROW_ROT=0`
- **作用**: Row rotation 模式
- **选项**:
  - `0` = 禁用 row rotation
  - `restore` = 恢复模式
  - `propagate` = 传播模式
- **说明**:
  - Row rotation 是一种提升量化精度的技术
  - 设置为0以简化推理流程

### 9. `OPENPI_DUQUANT_CALIB_STEPS=64`
- **作用**: Calibration（校准）步数
- **说明**:
  - 在量化前，用64个样本来估计激活值的分布
  - 更多步数 = 更准确的统计 = 更好的量化质量
  - 但会增加启动时间

## 性能 vs 精度权衡

### 当前配置（平衡型）
```bash
W4A8, BLOCK=64, PERMUTE=0, ROW_ROT=0
```
- **优点**: 较快的推理速度，中等内存占用
- **缺点**: 精度略低于无量化版本
- **适用**: 大多数评测场景

### 高精度配置
```bash
WBITS_DEFAULT=6, ABITS=8, BLOCK=32, PERMUTE=1, ROW_ROT=restore
```
- **优点**: 更高的精度
- **缺点**: 更慢的推理，更高内存占用

### 高速度配置
```bash
WBITS_DEFAULT=4, ABITS=8, BLOCK=128, PERMUTE=0, ROW_ROT=0
```
- **优点**: 最快的推理速度
- **缺点**: 精度可能略有下降

## 调试与验证

### 1. Dry-run 模式
查看哪些层会被量化，不实际替换：

```bash
OPENPI_DUQUANT_DEBUG=1 \
OPENPI_DUQUANT_DRYRUN=1 \
OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model." \
python examples/libero/main.py --headless ...
```

### 2. 查看量化层列表
```bash
python scripts/list_linears.py pi05_libero $CKPT \
  --scope paligemma_with_expert.gemma_expert.model.
```

### 3. 检查日志
在 headless 运行时，检查日志中的量化信息：

```
INFO:duquant:Quantizing layer: paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj
INFO:duquant:  Weight shape: [2048, 2048]
INFO:duquant:  Quantized to W4A8, block=64
INFO:duquant:  Memory saved: 12.5 MB
```

## 常见问题

### Q: 为什么推理速度变慢了？
A: 量化本身需要额外的量化/反量化开销。在某些硬件上，W4A8 可能比 FP16 更慢。但内存占用会显著降低。

### Q: 如何知道量化后精度损失多少？
A: 运行评测并比较成功率：
- 无量化: 运行不带 DuQuant 参数
- 有量化: 运行带完整 DuQuant 参数
- 比较两者的 success_rate

### Q: 可以只量化某些特定层吗？
A: 可以，使用更精确的 SCOPE 或 LAYERS 参数：

```bash
# 只量化 attention 层
OPENPI_DUQUANT_INCLUDE=".*attn.*"

# 排除某些层
OPENPI_DUQUANT_EXCLUDE=".*lm_head.*"

# 指定精确的层名
OPENPI_DUQUANT_LAYERS="layer.0.q_proj,layer.0.k_proj"
```

### Q: 为什么第一次推理很慢？
A: 第一次推理时需要：
1. 运行 calibration（64步）来估计激活值分布
2. 创建量化表
3. JIT/torch.compile 编译

后续推理会快得多。这就是为什么我们在评测前要做 warmup。

## 与 Headless 模式配合使用

完整命令示例：

```bash
# 设置环境
export CKPT=/path/to/checkpoint
export PYTHONPATH=$PWD/src:$PWD/third_party/libero
source examples/libero/.venv-libero/bin/activate

# 运行 headless 评测 + DuQuant
OPENPI_DUQUANT_DEBUG=1 \
OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model." \
OPENPI_DUQUANT_WBITS_DEFAULT=4 \
OPENPI_DUQUANT_ABITS=8 \
OPENPI_DUQUANT_ACT_PCT=98.0 \
OPENPI_DUQUANT_BLOCK=64 \
OPENPI_DUQUANT_PERMUTE=0 \
OPENPI_DUQUANT_ROW_ROT=0 \
OPENPI_DUQUANT_CALIB_STEPS=64 \
python examples/libero/main.py \
  --headless \
  --policy-config pi05_libero \
  --policy-dir "$CKPT" \
  --task-suite-name libero_spatial \
  --num-trials-per-task 20 \
  --seed 42
```

或使用辅助脚本（已包含完整参数）：

```bash
export CKPT=/path/to/checkpoint
bash examples/libero/run_headless_eval.sh
```

## 参考资料

- DuQuant 论文: [链接待补充]
- OpenPI README: [../../README.md](../../README.md)
- Headless 评测文档: [HEADLESS_EVALUATION.md](../../HEADLESS_EVALUATION.md)