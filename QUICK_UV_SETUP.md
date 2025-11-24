# OpenPI UV 环境快速设置

## ⚡ 一键安装（推荐）

```bash
cd ~/VLM_REPO/openpi
./setup_uv_env.sh
```

等待 10-15 分钟后，环境就准备好了！

## 🔧 使用环境

```bash
# 快速激活（使用自动生成的脚本）
source activate_env.sh

# 或手动激活
source .venv_test/bin/activate
export PYTHONPATH=$PWD/src:$PWD/third_party/libero
export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch

# 运行脚本
./examples/libero/run_quantvla.sh
```

## 📋 手动安装（3步）

### 1. 创建环境
```bash
cd ~/VLM_REPO/openpi
uv venv .venv_test --python 3.11
source .venv_test/bin/activate
```

### 2. 安装依赖
```bash
# OpenPI（包含所有依赖）
uv pip install -e .

# LIBERO
cd third_party/libero && uv pip install -e . && cd ../..

# BitBLAS（可选）
cd third_party/BitBLAS && uv pip install -e . && cd ../..
```

### 3. 验证
```bash
python -c "import torch, jax, libero, openpi; print('✓ OK')"
```

## 🎯 核心命令

| 操作 | 命令 |
|------|------|
| 创建环境 | `uv venv .venv_test --python 3.11` |
| 激活 | `source .venv_test/bin/activate` |
| 安装 OpenPI | `uv pip install -e .` |
| 安装 LIBERO | `uv pip install -e third_party/libero` |

## ❌ 常见错误

### 错误：openpi-client not found
✅ **解决**: 必须用 `uv pip install`，不能用 `pip install`

### 错误：CUDA not available
✅ **解决**:
```bash
nvcc --version  # 检查 CUDA
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu121
```

### 错误：uv command not found
✅ **解决**:
```bash
pip install uv
```

## ⏱️ 时间对比

| 方法 | 时间 |
|------|------|
| UV（这个方法） | 10-15 分钟 ⚡ |
| Conda | 30-60 分钟 🐌 |
| pip | 20-30 分钟 |

## 📦 安装的内容

自动从 `pyproject.toml` 安装约 40+ 个包：

**核心**:
- torch==2.7.1 (CUDA 12)
- jax[cuda12]==0.5.3
- transformers==4.53.2
- flax==0.10.2

**Workspace**:
- openpi-client (workspace package，只有 uv 支持)
- lerobot (from git)

**手动**:
- libero (from third_party)
- bitblas (from third_party, 可选)

## ✅ 检查清单

- [ ] uv 已安装: `uv --version`
- [ ] 环境已创建: `ls .venv_test/`
- [ ] 环境已激活: `echo $VIRTUAL_ENV`
- [ ] OpenPI 已安装: `python -c "import openpi"`
- [ ] CUDA 可用: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] LIBERO 可用: `python -c "import libero"`
- [ ] Checkpoint 存在: `ls ~/VLM_REPO/openpi/ckpts/pi05_libero_torch/`

全部通过 ✓ = 环境就绪！

## 🚀 下一步

```bash
# 先测试 FP baseline
./examples/libero/run_fp_baseline.sh

# 然后运行量化版本
./examples/libero/run_quantvla.sh
```

详细文档见: [UV_SETUP_GUIDE.md](UV_SETUP_GUIDE.md)
