# DuQuant SCOPE 配置快速参考

> **重要更新 (2025-10-01)**: 修复了INCLUDE regex以正确匹配Gemma的所有层
> - 添加了 `o_proj` (Gemma的attention输出投影)
> - 添加了 `gate_proj` (Gemma的MLP门控)
> - 现在每层匹配7个Linear（之前只匹配5个）
> - LLM和DiT现在各匹配 **126层** (之前是90层)

## 模型完整结构

```
PI0Pytorch
├── paligemma_with_expert (PaliGemmaWithExpertModel)
│   ├── paligemma (PaliGemmaForConditionalGeneration)
│   │   └── model (PaliGemmaModel)
│   │       ├── vision_tower (SigLIPVisionModel)
│   │       │   └── vision_model.encoder.layers[0-26]
│   │       │       ├── self_attn.{q_proj, k_proj, v_proj, out_proj}
│   │       │       └── mlp.{fc1, fc2}
│   │       ├── language_model (GemmaModel)
│   │       │   └── layers[0-17]
│   │       │       ├── self_attn.{q_proj, k_proj, v_proj, o_proj}
│   │       │       └── mlp.{gate_proj, up_proj, down_proj}
│   │       └── multi_modal_projector (PaliGemmaMultiModalProjector)
│   │           └── linear
│   └── gemma_expert (GemmaForCausalLM)
│       └── model (GemmaModel)
│           └── layers[0-17]
│               ├── self_attn.{q_proj, k_proj, v_proj, o_proj}
│               └── mlp.{gate_proj, up_proj, down_proj}
├── action_in_proj (Linear)
├── action_out_proj (Linear)
└── time_mlp_in/time_mlp_out (Linear, 仅pi0.5)
```

---

## SCOPE配置速查表

| 目标组件 | SCOPE值 | 层数估计 | 脚本 |
|---------|---------|---------|------|
| **DiT (动作专家)** | `paligemma_with_expert.gemma_expert.model.` | ~126 | `run_optimized_duquant.sh` |
| **LLM (语言模型)** | `paligemma_with_expert.paligemma.model.language_model.` | ~126 | `run_llm_w4a8.sh` |
| **Vision (视觉编码器)** | `paligemma_with_expert.paligemma.model.vision_tower.` | ~108 | (需自定义) |
| **PaliGemma全部** | `paligemma_with_expert.paligemma.model.` | ~234 | (需自定义) |
| **全模型** | `paligemma_with_expert.` | ~360+ | `run_full_quantize.sh` |

---

## 常用配置示例

### 1. 量化DiT (默认/当前配置)

```bash
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model."
export OPENPI_DUQUANT_PACKDIR="duquant_packed_dit_w4a8"
bash examples/libero/run_optimized_duquant.sh
```

**匹配的层示例**:
- `paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj`
- `paligemma_with_expert.gemma_expert.model.layers.0.mlp.gate_proj`
- ...

---

### 2. 量化LLM (推荐 - 最大内存节省)

```bash
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.model.language_model."
export OPENPI_DUQUANT_PACKDIR="duquant_packed_llm_w4a8"
bash examples/libero/run_llm_w4a8.sh
```

**匹配的层示例**:
- `paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj`
- `paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.gate_proj`
- ...

---

### 3. 量化Vision (视觉编码器)

```bash
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.model.vision_tower."
export OPENPI_DUQUANT_PACKDIR="duquant_packed_vision_w4a8"
```

**匹配的层示例**:
- `paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj`
- `paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.mlp.fc1`
- ...

---

### 4. 量化LLM + DiT (不含Vision)

**方法A - 使用EXCLUDE**:
```bash
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert."
export OPENPI_DUQUANT_EXCLUDE="vision_tower"
export OPENPI_DUQUANT_PACKDIR="duquant_packed_llm_dit_w4a8"
```

**方法B - 两次运行**:
```bash
# 第一次量化LLM
bash examples/libero/run_llm_w4a8.sh

# 然后在main.py中量化DiT (需要代码修改)
```

---

### 5. 量化所有Transformer组件

```bash
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert."
export OPENPI_DUQUANT_PACKDIR="duquant_packed_full_w4a8"
bash examples/libero/run_full_quantize.sh
```

**匹配的层**: Vision + LLM + DiT 全部

---

## 重要说明

### ⚠️ 关键路径差异

**为什么DiT是 `.gemma_expert.model.` 而LLM是 `.paligemma.model.language_model.`？**

因为：
1. **GemmaForCausalLM** (DiT):
   ```python
   gemma_expert.model.layers[0-17]
   ```

2. **PaliGemmaForConditionalGeneration** (Vision+LLM):
   ```python
   paligemma.model.language_model.layers[0-17]  # LLM
   paligemma.model.vision_tower.vision_model... # Vision
   ```

**PaliGemma内部有一个额外的 `.model` 层级！**

---

### 验证SCOPE是否正确

使用dry-run模式测试：

```bash
export OPENPI_DUQUANT_DEBUG=1
export OPENPI_DUQUANT_DRYRUN=1
export OPENPI_DUQUANT_SCOPE="your_scope_here"

python examples/libero/main.py --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1
```

输出会显示：
- `[DUQUANT] SCOPE filter: 'your_scope_here'`
- `[DUQUANT] Matched Linear layers: X`
- 如果X=0，会打印DEBUG信息显示实际的层名称

---

## 常见错误

### ❌ 错误 1: 漏掉 `.model.`

```bash
# 错误
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.language_model."

# 正确
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.model.language_model."
```

**结果**: 匹配0层！

---

### ❌ 错误 2: PACKDIR不匹配配置

```bash
# 使用LLM的SCOPE但用了DiT的PACKDIR
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.model.language_model."
export OPENPI_DUQUANT_PACKDIR="duquant_packed_dit_w4a8"  # 错误！
```

**结果**: Packdir包含错误的rotation matrices，可能导致数值错误！

**每个不同的配置(PERMUTE/ROW_ROT/BLOCK等)必须使用独立的PACKDIR！**

---

### ❌ 错误 3: 忘记末尾的点

```bash
# 可能匹配不到或匹配错误的层
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.model.language_model"

# 正确 - 末尾带点确保只匹配子模块
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.model.language_model."
```

---

### ✅ 2025-10-01 修复: INCLUDE regex更新

**之前的问题** (已修复):
- 默认INCLUDE regex缺少 `o_proj` 和 `gate_proj`
- 每层只匹配5个Linear (q_proj, k_proj, v_proj, up_proj, down_proj)
- 漏掉了2个重要层

**修复后**:
```python
# src/openpi/models_pytorch/duquant_layers.py:392
INCLUDE = r".*(q_proj|k_proj|v_proj|o_proj|out_proj|fc1|fc2|gate_proj|up_proj|down_proj).*"
```

**现在匹配全部7个Linear**:
- Attention: `q_proj, k_proj, v_proj, o_proj` (4个) ✅
- MLP: `gate_proj, up_proj, down_proj` (3个) ✅

**影响**:
- 更完整的量化覆盖
- 内存节省从 ~25% 提升到 ~35%
- 略微增加精度损失，但更符合预期

---

## 调试技巧

### 1. 检查模型结构

```python
import torch
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import TrainConfig

# 加载模型
config = TrainConfig.load('pi05_libero')
policy = create_trained_policy(config, '/path/to/ckpt', pytorch_device='cpu')

# 打印所有Linear层
for name, module in policy.model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)
```

### 2. 使用DEBUG模式

在 `duquant_layers.py` 中添加了调试输出，当匹配0层时会自动打印：
- 模型中总共的Linear层数
- 前10个Linear层的名称
- 匹配SCOPE前缀的层数量

---

## 快速决策树

```
需要量化哪个组件？
├─ 内存最紧张？
│  └─ 量化LLM (最大单组件，~40%节省)
│     SCOPE="paligemma_with_expert.paligemma.model.language_model."
│
├─ 保持语言理解精度？
│  └─ 量化DiT (当前默认，~20%节省)
│     SCOPE="paligemma_with_expert.gemma_expert.model."
│
├─ 视觉任务不重要？
│  └─ 量化Vision (~15%节省)
│     SCOPE="paligemma_with_expert.paligemma.model.vision_tower."
│
└─ 极限压缩？
   └─ 量化全部 (~60%节省，精度损失)
      SCOPE="paligemma_with_expert."
```

---

## 参考文档

- 完整脚本指南: `DUQUANT_SCRIPTS_GUIDE.md`
- 配置详解: `DUQUANT_CONFIG.md`
- 优化历史: `DUQUANT_OPTIMIZATION_CHANGELOG.md`
