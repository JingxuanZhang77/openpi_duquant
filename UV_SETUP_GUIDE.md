# OpenPI UV 虚拟环境 Setup 指南

## ⚡ 使用 UV 创建新环境（正确方法）

OpenPI 使用 **uv** 作为包管理器，而不是 conda。

## 🚀 快速开始

```bash
cd ~/VLM_REPO/openpi

# 创建新的 uv 虚拟环境
uv venv .venv_test --python 3.11

# 激活环境
source .venv_test/bin/activate

# 安装 OpenPI 及所有依赖
uv pip install -e .

# 安装 LIBERO
cd third_party/libero
uv pip install -e .
cd ../..

# 安装 BitBLAS（如果需要）
cd third_party/BitBLAS
uv pip install -e .
cd ../..

# 完成！
```

## 📋 完整步骤详解

### 1. 安装 uv（如果还没有）

```bash
# 使用 pip 安装
pip install uv

# 或使用官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 创建虚拟环境

```bash
cd ~/VLM_REPO/openpi

# 使用 uv 创建 Python 3.11 环境
uv venv .venv_test --python 3.11
```

这会在当前目录创建 `.venv_test/` 文件夹。

### 3. 激活环境

```bash
source .venv_test/bin/activate
```

你会看到命令行提示符前面出现 `(.venv_test)`。

### 4. 安装 OpenPI

```bash
# uv 会自动解析 pyproject.toml 并安装所有依赖
uv pip install -e .
```

这会安装：
- torch==2.7.1
- jax[cuda12]==0.5.3
- transformers==4.53.2
- flax==0.10.2
- openpi-client (workspace package)
- lerobot (from git)
- 以及所有其他依赖...

### 5. 安装 LIBERO

```bash
cd third_party/libero
uv pip install -e .
cd ../..
```

### 6. 安装 BitBLAS（可选）

```bash
cd third_party/BitBLAS
uv pip install -e .
cd ../..
```

### 7. 验证安装

```bash
python -c "
import torch
import jax
import libero
import openpi
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA:', torch.cuda.is_available())
print('✓ JAX:', jax.__version__)
print('✓ LIBERO: OK')
print('✓ OpenPI: OK')
print('\\nEnvironment ready!')
"
```

## 🎯 运行脚本

```bash
# 设置环境变量
export PYTHONPATH=$PWD/src:$PWD/third_party/libero
export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch

# 运行 quantvla 脚本
./examples/libero/run_quantvla.sh
```

## 🔍 UV vs Conda

| 特性 | UV | Conda |
|------|-----|-------|
| 速度 | ⚡ 10-100x 更快 | 慢 |
| 依赖解析 | 智能，快速 | 慢，有时失败 |
| pyproject.toml | ✅ 原生支持 | ❌ 需要额外配置 |
| workspace | ✅ 支持 | ❌ 不支持 |
| 磁盘占用 | 小 | 大 |

OpenPI 的 `pyproject.toml` 使用了 workspace 和 git dependencies，这些只有 uv 能正确处理。

## 📦 关键依赖

### 从 pyproject.toml 自动安装
```toml
[project]
dependencies = [
    "torch==2.7.1",
    "jax[cuda12]==0.5.3",
    "transformers==4.53.2",
    "flax==0.10.2",
    ... 等40+个包
]

[tool.uv.sources]
openpi-client = { workspace = true }  # 这个 conda 不支持！
lerobot = { git = "https://github.com/..." }  # 这个也需要 uv
```

### 手动安装
- LIBERO (from `third_party/libero`)
- BitBLAS (from `third_party/BitBLAS`, 可选)

## ⏱️ 时间估算

| 步骤 | 时间 |
|------|------|
| 安装 uv | 30秒 |
| 创建虚拟环境 | 10秒 |
| 安装 OpenPI | 5-10分钟 |
| 安装 LIBERO | 1-2分钟 |
| 安装 BitBLAS | 2-3分钟 |
| **总计** | **10-15分钟** |

比 conda 快很多！

## 🆘 常见问题

### 问题：uv 找不到

```bash
# 安装 uv
pip install uv
# 或
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 问题：CUDA 不可用

```bash
# 检查系统 CUDA
nvcc --version
nvidia-smi

# uv 安装的 torch 应该自带 CUDA support
# 如果不行，手动指定 CUDA 版本：
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu121
```

### 问题：openpi-client 找不到

这是正常的！`openpi-client` 是 workspace package，必须用 uv。

如果你用 `pip install -e .` 而不是 `uv pip install -e .`，会失败。

### 问题：lerobot 安装失败

```bash
# uv 会自动从 git 安装
# 如果失败，手动安装：
uv pip install git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
```

## 🔄 多个环境管理

你可以创建多个测试环境：

```bash
# 测试环境 1
uv venv .venv_test1 --python 3.11

# 测试环境 2
uv venv .venv_test2 --python 3.11

# 激活不同环境
source .venv_test1/bin/activate  # 或
source .venv_test2/bin/activate
```

## 📊 环境对比

| 环境 | 路径 | 用途 | Python | 管理器 |
|------|------|------|--------|--------|
| 现有工作环境 | `examples/libero/.venv` | 你的主环境 | 3.11.13 | uv 0.8.20 |
| 新测试环境 | `.venv_test` | 测试用 | 3.11 | uv 0.8.20 |

## 💡 UV 优势

1. **快速**: 比 pip/conda 快 10-100 倍
2. **准确**: 依赖解析更智能
3. **现代**: 原生支持 pyproject.toml 和 workspace
4. **兼容**: 完全兼容 pip
5. **简单**: 命令与 pip 相似

## ✅ 完整安装验证

```bash
# 在新环境中运行
python << 'EOF'
import sys
import torch
import jax
import transformers
import flax
import libero
import openpi

print("="*60)
print("UV Environment Verification")
print("="*60)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
print(f"JAX: {jax.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Flax: {flax.__version__}")
print("LIBERO: ✓")
print("OpenPI: ✓")

from libero.libero import benchmark
bench = benchmark.get_benchmark_dict()['libero_10']()
print(f"LIBERO Benchmark: {bench.n_tasks} tasks")

print("="*60)
print("All packages verified! ✓")
print("="*60)
EOF
```

## 🎁 Bonus: UV 常用命令

```bash
# 创建环境
uv venv .venv --python 3.11

# 安装包
uv pip install package_name

# 安装 editable 包
uv pip install -e .

# 列出已安装的包
uv pip list

# 冻结依赖
uv pip freeze > requirements.txt

# 从 requirements 安装
uv pip install -r requirements.txt

# 升级 uv 本身
uv self update
```

## 📚 参考

- UV 官方文档: https://github.com/astral-sh/uv
- OpenPI pyproject.toml: 查看项目根目录

## 🎯 下一步

环境创建完成后：

1. 运行 FP baseline: `./examples/libero/run_fp_baseline.sh`
2. 运行 QuantVLA: `./examples/libero/run_quantvla.sh`
3. 对比结果

祝实验顺利！🚀
