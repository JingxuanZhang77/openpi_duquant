# OpenPI LIBERO环境快速部署指南

## 一键安装（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/physical-intelligence/openpi.git
cd openpi

# 2. 运行自动化安装脚本（10-15分钟）
bash examples/libero/setup_environment.sh

# 3. 激活环境
source examples/libero/activate_env.sh

# 4. 开始实验
bash examples/libero/run_optimized_duquant.sh
```

## 环境规格

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.11.13 | 由uv管理 |
| PyTorch | 2.7.1 | CUDA 12.4支持 |
| JAX | 0.5.3 | CUDA 12支持 |
| Transformers | 4.53.2 | HuggingFace模型 |
| CUDA | 12.4+ | GPU加速 |

## 已安装的脚本

环境安装后，你可以运行以下脚本：

### 1. DiT全量化（126层）
```bash
source examples/libero/activate_env.sh
bash examples/libero/run_optimized_duquant.sh
```

### 2. DiT注意力层量化（72层）
```bash
source examples/libero/activate_env.sh
bash examples/libero/run_dit_qkvo_w4a8.sh
```

### 3. LLM量化（126层）
```bash
source examples/libero/activate_env.sh
bash examples/libero/run_llm_w4a8.sh
```

### 4. LLM+DiT全量化（252层）
```bash
source examples/libero/activate_env.sh
bash examples/libero/run_full_llm_dit_w4a8.sh
```

## DuQuant配置

所有脚本已预配置：
- ✅ W4A8 fake quantization
- ✅ Block-wise rotation（block_size=16）
- ✅ Permutation enabled
- ✅ Output restoration (ROW_ROT=restore)
- ✅ Batched rotation optimization（减少GPU kernel launch）
- ✅ Expandable segments（避免OOM）

## 系统要求

### 最低配置
- Ubuntu 20.04+
- NVIDIA GPU (12GB+ VRAM)
- 16GB RAM
- 30GB 磁盘空间

### 推荐配置
- Ubuntu 22.04/24.04
- NVIDIA A40/A100 (40GB+ VRAM)
- 32GB+ RAM
- 50GB 磁盘空间

## 快速验证

```bash
source examples/libero/activate_env.sh

python -c "
import torch, jax, transformers, openpi, libero
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA:', torch.cuda.is_available())
print('✓ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('✓ JAX:', jax.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ OpenPI & LIBERO: installed')
"
```

## 常见问题

### Q: OOM错误？
A: 已配置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

如果仍然OOM，可以禁用批量旋转优化：
```bash
export OPENPI_DUQUANT_BATCH_ROT=0
```

### Q: uv找不到？
A: 添加到PATH：
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

### Q: CUDA不可用？
A: 检查NVIDIA驱动：
```bash
nvidia-smi
```

重装PyTorch：
```bash
uv pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

## 文件结构

```
openpi/
├── examples/libero/
│   ├── setup_environment.sh      # 自动化安装脚本
│   ├── activate_env.sh            # 环境激活脚本（自动生成）
│   ├── run_optimized_duquant.sh   # DiT全量化
│   ├── run_dit_qkvo_w4a8.sh       # DiT注意力层量化
│   ├── run_llm_w4a8.sh            # LLM量化
│   ├── run_full_llm_dit_w4a8.sh   # LLM+DiT全量化
│   ├── .venv/                     # Python虚拟环境
│   └── requirements.txt           # Python依赖
├── src/openpi/
│   └── models_pytorch/
│       ├── duquant_layers.py      # DuQuant主要实现
│       └── duquant_preprocess.py  # DuQuant预处理
└── third_party/
    ├── libero/                    # LIBERO基准测试
    └── BitBLAS/                   # 可选：int4/int8加速
```

## 性能预期

### 显存占用
- **FP16 baseline**: ~34 GB
- **DiT W4A8 (126层)**: ~35 GB (+1 GB)
- **LLM+DiT W4A8 (252层)**: ~36 GB (+2 GB)

### 速度（DiT量化）
- **FP16 baseline**: ~2 秒/policy call
- **W4A8 fake quant (无优化)**: ~13 秒/policy call
- **W4A8 fake quant (批量旋转)**: ~4-6 秒/policy call（预期）

**注意**：fake quantization不会加速，需要实现真实int4/int8 kernel才能获得加速。

## 迁移到新服务器

1. 拷贝脚本到新服务器：
```bash
scp examples/libero/setup_environment.sh new_server:~/
```

2. SSH到新服务器运行：
```bash
ssh new_server
bash setup_environment.sh
```

3. 一切就绪！

## 更新日志

- **2025-10-12**: 初始版本
  - 支持Python 3.11
  - PyTorch 2.7.1 + CUDA 12
  - JAX 0.5.3
  - DuQuant批量旋转优化
  - Expandable segments支持

## 联系方式

如有问题，请查看：
- [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) - 完整文档
- [OpenPI GitHub](https://github.com/physical-intelligence/openpi)
- [LIBERO Project](https://libero-project.github.io/)
