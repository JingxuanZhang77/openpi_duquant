# DuQuant Quantization Scripts Guide

本指南说明如何使用不同的DuQuant量化脚本来优化OpenPI模型的不同组件。

## 模型结构

OpenPI (PI0Pytorch) 模型包含三个主要的Transformer组件：

```
PI0Pytorch
├── paligemma_with_expert.paligemma
│   ├── vision_tower (SigLIP视觉编码器)
│   │   └── 27层Transformer × 4个Linear层 = ~108个线性层
│   └── language_model (Gemma语言模型/LLM)
│       └── 18层Transformer × 7个Linear层 = ~126个线性层
└── paligemma_with_expert.gemma_expert (DiT动作专家)
    └── model
        └── 18层Transformer × 7个Linear层 = ~126个线性层

总计：~360个线性层
```

---

## 可用脚本对比

| 脚本 | 量化目标 | 层数 | 内存节省 | 精度影响 | 推荐场景 |
|------|---------|------|---------|---------|---------|
| `run_simple_w4a8.sh` | DiT | ~126 | ~20% | 较大 | 快速测试 |
| `run_optimized_duquant.sh` | DiT | ~126 | ~20% | 中等 | DiT量化（完整优化） |
| `run_llm_w4a8.sh` | LLM | ~126 | ~40% | 中等 | **推荐**：最大单组件节省 |
| `run_full_quantize.sh` | 全部 | ~360 | ~60% | 较大 | 极限内存节省 |

---

## 1. run_simple_w4a8.sh - 简化DiT量化

**量化范围：** 只量化DiT (动作生成)
**优化：** 禁用permutation和rotation（最快）

### 配置
```bash
SCOPE="paligemma_with_expert.gemma_expert.model."
PERMUTE=0        # 禁用排列
ROW_ROT=0        # 禁用旋转
torch.compile=禁用
```

### 使用场景
- 快速验证量化是否工作
- 不需要最高精度
- 追求最快的启动和运行速度

### 运行
```bash
bash examples/libero/run_simple_w4a8.sh
```

### 预期性能
- 速度：~1-2分钟/episode
- 内存节省：~20%
- 精度：较低（无permutation/rotation优化）

---

## 2. run_optimized_duquant.sh - 完整DiT量化

**量化范围：** 只量化DiT (动作生成)
**优化：** 启用permutation和rotation（更高精度）

### 配置
```bash
SCOPE="paligemma_with_expert.gemma_expert.model."
PERMUTE=1        # 启用排列
ROW_ROT=restore  # 启用旋转恢复
torch.compile=禁用（可改）
```

### 使用场景
- 需要更高精度的DiT量化
- 完整DuQuant优化
- 保持LLM全精度

### 运行
```bash
bash examples/libero/run_optimized_duquant.sh
```

### 预期性能
- 速度：~2-3分钟/episode (torch.compile禁用)
- 内存节省：~20%
- 精度：中等（有permutation/rotation优化）

### 启用torch.compile（推荐）
编辑脚本，将以下行改为：
```bash
unset OPENPI_DISABLE_TORCH_COMPILE
unset TORCH_COMPILE_DISABLE
unset TORCHDYNAMO_DISABLE
```
- 第一个episode：~15-20分钟（编译）
- 后续episode：~30-60秒

---

## 3. run_llm_w4a8.sh - LLM量化 ⭐推荐

**量化范围：** 只量化LLM (语言理解)
**优化：** 启用permutation和rotation

### 配置
```bash
SCOPE="paligemma_with_expert.paligemma.language_model."
PERMUTE=1        # 启用排列
ROW_ROT=restore  # 启用旋转恢复
torch.compile=禁用（可改）
```

### 使用场景
- **推荐首选！** LLM是最大组件
- 最大单组件内存节省
- 保持DiT和视觉全精度

### 运行
```bash
bash examples/libero/run_llm_w4a8.sh
```

### 预期性能
- 速度：~2-3分钟/episode
- 内存节省：~40%（最大单组件节省）
- 精度：中等

### 为什么推荐量化LLM？
1. LLM占用最多内存（~126个线性层，每层都很大）
2. 语言理解对量化误差相对鲁棒
3. 保持DiT全精度对动作生成精度更重要

---

## 4. run_full_quantize.sh - 全模型量化

**量化范围：** 量化所有组件（Vision + LLM + DiT）
**优化：** 启用permutation和rotation

### 配置
```bash
SCOPE="paligemma_with_expert."  # 不指定子模块，量化所有
PERMUTE=1
ROW_ROT=restore
torch.compile=禁用
```

### 使用场景
- 极限内存节省
- 内存严重受限的环境
- 可以接受一定精度损失

### 运行
```bash
bash examples/libero/run_full_quantize.sh
```
（脚本会要求确认，因为这是最激进的量化）

### 预期性能
- 速度：~3-5分钟/episode
- 内存节省：~60%（最大可能）
- 精度：可能明显下降

### ⚠️ 警告
- 量化~360个线性层
- 第一次运行需要生成大量pack文件（5-10分钟）
- 如果精度太低，改用LLM或DiT单独量化

---

## 自定义量化配置

### 只量化视觉编码器
```bash
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.vision_tower."
```

### 量化LLM和DiT，但不量化视觉
```bash
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert."
export OPENPI_DUQUANT_EXCLUDE="vision_tower"
```

### 只量化注意力层（所有组件）
```bash
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert."
export OPENPI_DUQUANT_INCLUDE=".*(q_proj|k_proj|v_proj|o_proj|out_proj).*"
```

### 只量化MLP层（所有组件）
```bash
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert."
export OPENPI_DUQUANT_INCLUDE=".*(fc1|fc2|gate_proj|up_proj|down_proj).*"
```

---

## 验证量化配置（Dry Run）

在实际运行前，可以使用dry-run模式查看哪些层会被量化：

```bash
export OPENPI_DUQUANT_DEBUG=1
export OPENPI_DUQUANT_DRYRUN=1
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.language_model."

python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1
```

输出会列出所有匹配的层，但不会实际替换。

---

## 推荐工作流程

### 1. 首次测试（快速验证）
```bash
bash examples/libero/run_simple_w4a8.sh
```
验证量化基本功能，获得baseline性能。

### 2. LLM量化（推荐）
```bash
bash examples/libero/run_llm_w4a8.sh
```
获得最大内存节省，评估精度影响。

### 3. 对比DiT量化
```bash
bash examples/libero/run_optimized_duquant.sh
```
对比LLM vs DiT量化的效果差异。

### 4. （可选）全模型量化
```bash
bash examples/libero/run_full_quantize.sh
```
如果内存严重不足且精度可接受。

---

## 参数调优指南

### 提升精度
```bash
export OPENPI_DUQUANT_WBITS_DEFAULT=8  # 从4位提升到8位
export OPENPI_DUQUANT_PERMUTE=1        # 启用排列
export OPENPI_DUQUANT_ROW_ROT=restore  # 启用旋转
export OPENPI_DUQUANT_BLOCK=32         # 增大block size
```

### 提升速度
```bash
export OPENPI_DUQUANT_PERMUTE=0        # 禁用排列
export OPENPI_DUQUANT_ROW_ROT=0        # 禁用旋转
export OPENPI_DUQUANT_CALIB_STEPS=16   # 减少校准步数
```

### 平衡精度和速度
```bash
export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_PERMUTE=1
export OPENPI_DUQUANT_ROW_ROT=restore
export OPENPI_DUQUANT_BLOCK=16
export OPENPI_DUQUANT_CALIB_STEPS=32
```
（这是当前脚本的默认配置）

---

## 常见问题

### Q1: 量化后成功率为0%怎么办？
**A:** 尝试以下方法：
1. 增加权重位数：`WBITS_DEFAULT=8`
2. 增加激活位数：`ABITS=16`
3. 减少量化范围：只量化LLM或DiT，不要全量化
4. 检查packdir是否与配置匹配

### Q2: torch.compile很慢怎么办？
**A:** 这是正常的：
- 第一次运行需要15-20分钟编译
- 后续运行使用缓存，只需30-60秒
- 如果只运行一次，建议禁用torch.compile

### Q3: 如何选择量化哪个组件？
**A:** 推荐顺序：
1. **LLM** (最大内存节省，中等精度影响)
2. **DiT** (中等内存节省，可能影响动作精度)
3. **Vision** (较小内存节省，可能影响视觉理解)
4. **全部** (最大节省，精度影响最大)

### Q4: packdir冲突怎么办？
**A:** 每个配置必须使用独立的packdir：
- DiT简单版：`duquant_packed_w4a8_simple`
- DiT完整版：`duquant_packed_b16_p1_rrestore_a999`
- LLM：`duquant_packed_llm_w4a8`
- 全量化：`duquant_packed_full_w4a8`

不同的PERMUTE/ROW_ROT配置不能共用packdir！

---

## 技术细节

### DuQuant环境变量完整列表

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENPI_DUQUANT_SCOPE` | `"policy.dit."` | 量化范围（必须设置！） |
| `OPENPI_DUQUANT_WBITS_DEFAULT` | `4` | 权重位数 |
| `OPENPI_DUQUANT_ABITS` | `8` | 激活位数 |
| `OPENPI_DUQUANT_BLOCK` | `16` | Block size |
| `OPENPI_DUQUANT_PERMUTE` | `1` | 是否启用排列 |
| `OPENPI_DUQUANT_ROW_ROT` | `"restore"` | 行旋转模式 |
| `OPENPI_DUQUANT_ACT_PCT` | `99.9` | 激活百分位 |
| `OPENPI_DUQUANT_CALIB_STEPS` | `32` | 校准步数 |
| `OPENPI_DUQUANT_LS` | `0.15` | Lambda smooth |
| `OPENPI_DUQUANT_PACKDIR` | `None` | Pack文件目录 |
| `OPENPI_DUQUANT_DEBUG` | `0` | 调试输出 |
| `OPENPI_DUQUANT_DRYRUN` | `0` | 只列出层，不替换 |

### SCOPE值参考

```bash
# DiT (动作专家 - Action Expert)
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model."

# LLM (语言模型 - Gemma Language Model)
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.model.language_model."

# Vision (视觉编码器 - SigLIP Vision Tower)
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.model.vision_tower."

# 全部组件 (Vision + LLM + DiT)
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert."

# 注意：PaliGemma内部有一个.model层级，所以路径是：
#   paligemma_with_expert.paligemma.model.{language_model|vision_tower|multi_modal_projector}
```

---

## 联系与反馈

如有问题或需要帮助，请参考：
- `DUQUANT_CONFIG.md` - 详细配置说明
- `DUQUANT_OPTIMIZATION_CHANGELOG.md` - 优化历史
- `README.md` - 总体使用指南
