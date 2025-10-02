# Headless 模式环境配置

## 问题说明

Headless 模式需要在同一进程中加载 policy，因此需要同时安装：
1. **LIBERO 依赖**（环境模拟）
2. **OpenPI 依赖**（policy 推理：JAX/PyTorch, transformers 等）

如果只安装了 LIBERO 依赖，会遇到错误：
```
ModuleNotFoundError: No module named 'jax'
```

## 解决方案

### 方案 1：在现有环境中安装 OpenPI 依赖（推荐）

```bash
# 激活 LIBERO 环境
source examples/libero/.venv-libero/bin/activate

# 安装 OpenPI 核心依赖
pip install -e .  # 从项目根目录安装 openpi

# 或者手动安装关键依赖
pip install jax[cuda12] flax torch transformers pillow
```

### 方案 2：创建完整的评测环境

创建一个包含所有依赖的新环境：

```bash
# 1. 创建新环境
python3.9 -m venv examples/libero/.venv-headless
source examples/libero/.venv-headless/bin/activate

# 2. 升级 pip
pip install --upgrade pip

# 3. 安装 LIBERO 依赖
pip install -r examples/libero/requirements.txt \
            -r third_party/libero/requirements.txt \
            --extra-index-url https://download.pytorch.org/whl/cu113

# 4. 安装 OpenPI 依赖
pip install -e .  # 从项目根目录

# 5. 安装 openpi-client
pip install -e packages/openpi-client

# 6. 其他必需包
pip install sentencepiece

# 7. （可选）设置 robosuite 宏
python $(python -c "import robosuite, pathlib; \
    print(pathlib.Path(robosuite.__file__).parent / 'scripts' / 'setup_macros.py')")
```

### 方案 3：使用 Docker（最可靠）

Docker 环境已经包含所有依赖：

```bash
# 授予 X11 访问权限
sudo xhost +local:docker

# 运行 headless 评测
SERVER_ARGS="--env LIBERO" \
CLIENT_ARGS="--args.headless --args.policy-config pi05_libero --args.policy-dir /path/to/checkpoint" \
docker compose -f examples/libero/compose.yml up --build
```

注意：需要修改 `compose.yml` 以支持 headless 模式。

## 推荐配置（方案 1 详细步骤）

如果你已经有 `.venv-libero` 环境，最简单的方法是补充安装 OpenPI 依赖：

```bash
# 1. 激活环境
source examples/libero/.venv-libero/bin/activate

# 2. 检查 Python 版本（应该是 3.9）
python --version

# 3. 从项目根目录安装 OpenPI
cd ~/VLM_REPO/openpi
pip install -e .

# 4. 验证安装
python -c "import jax; print('JAX version:', jax.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "from openpi.policies import policy_config; print('OpenPI imported successfully')"

# 5. 设置 PYTHONPATH
export PYTHONPATH=$PWD/src:$PWD/third_party/libero

# 6. 运行 headless 评测
export CKPT=/path/to/checkpoint
bash examples/libero/run_headless_eval.sh
```

## 依赖列表

Headless 模式需要以下关键依赖：

### LIBERO 相关
- `robosuite`
- `mujoco`
- `libero` (from third_party)

### OpenPI 相关
- `jax` / `jaxlib` (JAX 模型)
- `torch` (PyTorch 模型)
- `flax` (JAX 训练)
- `transformers` (HuggingFace 模型)
- `pillow` (图像处理)
- `sentencepiece` (tokenizer)
- `numpy`, `scipy` 等基础库

### DuQuant 相关（如果使用量化）
DuQuant 通常已经包含在 OpenPI 中，如果单独需要：
- 检查 `src/openpi/models/` 下是否有 duquant 相关代码

## 验证环境配置

运行以下命令检查环境是否正确配置：

```bash
# 激活环境
source examples/libero/.venv-libero/bin/activate
export PYTHONPATH=$PWD/src:$PWD/third_party/libero

# 测试 LIBERO
python -c "from libero.libero import benchmark; print('LIBERO OK')"

# 测试 OpenPI
python -c "from openpi.policies import policy_config; print('OpenPI OK')"

# 测试 JAX
python -c "import jax; print('JAX version:', jax.__version__)"

# 测试 PyTorch
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# 如果都成功，说明环境配置正确
```

## 常见问题

### Q1: `ModuleNotFoundError: No module named 'jax'`

**原因**: 当前环境缺少 JAX

**解决**:
```bash
pip install jax[cuda12] jaxlib
# 或者针对特定 CUDA 版本
pip install jax[cuda11_cudnn86]
```

### Q2: `ModuleNotFoundError: No module named 'openpi'`

**原因**: 没有安装 OpenPI 包

**解决**:
```bash
# 从项目根目录
pip install -e .
```

### Q3: PyTorch 版本冲突

**原因**: LIBERO 依赖可能指定了旧版本 PyTorch

**解决**:
```bash
# 先安装 LIBERO 依赖，再升级 PyTorch
pip install -r examples/libero/requirements.txt
pip install --upgrade torch torchvision
```

### Q4: CUDA 版本不匹配

**原因**: JAX/PyTorch 的 CUDA 版本与系统不匹配

**检查系统 CUDA 版本**:
```bash
nvidia-smi
# 查看 "CUDA Version: XX.X"
```

**安装对应版本**:
```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install jax[cuda11_cudnn86]

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install jax[cuda12]
```

## 快速检查清单

在运行 headless 评测前，确保：

- [ ] 激活了正确的虚拟环境
- [ ] 设置了 `PYTHONPATH`
- [ ] 可以 `import jax`
- [ ] 可以 `import torch`
- [ ] 可以 `from openpi.policies import policy_config`
- [ ] 可以 `from libero.libero import benchmark`
- [ ] 设置了 `CKPT` 环境变量
- [ ] checkpoint 路径存在且正确

## 完整安装示例

从零开始创建 headless 评测环境：

```bash
#!/bin/bash
set -e

# 1. 进入项目目录
cd ~/VLM_REPO/openpi

# 2. 创建虚拟环境
python3.9 -m venv examples/libero/.venv-headless
source examples/libero/.venv-headless/bin/activate

# 3. 升级 pip
pip install --upgrade pip

# 4. 安装 LIBERO 依赖
pip install -r examples/libero/requirements.txt \
            -r third_party/libero/requirements.txt \
            --extra-index-url https://download.pytorch.org/whl/cu113 \
            --index-strategy=unsafe-best-match

# 5. 安装 OpenPI（从根目录）
pip install -e .

# 6. 安装 openpi-client
pip install -e packages/openpi-client

# 7. 安装额外依赖
pip install sentencepiece

# 8. 验证安装
echo "Verifying installation..."
python -c "import jax; print('✓ JAX:', jax.__version__)"
python -c "import torch; print('✓ PyTorch:', torch.__version__)"
python -c "from openpi.policies import policy_config; print('✓ OpenPI')"
python -c "from libero.libero import benchmark; print('✓ LIBERO')"

echo "✅ Environment setup complete!"
echo "To use: source examples/libero/.venv-headless/bin/activate"
```

保存为 `examples/libero/setup_headless_env.sh` 并运行：

```bash
bash examples/libero/setup_headless_env.sh
```

## 运行评测

环境配置好后：

```bash
# 激活环境
source examples/libero/.venv-headless/bin/activate

# 设置路径
export PYTHONPATH=$PWD/src:$PWD/third_party/libero
export CKPT=/path/to/checkpoint

# 运行评测
bash examples/libero/run_headless_eval.sh
```

## 相关文档

- [HEADLESS_QUICKSTART.md](HEADLESS_QUICKSTART.md) - 快速开始指南
- [README.md](README.md) - LIBERO 总体说明
- [PARAMETER_FIX.md](PARAMETER_FIX.md) - 参数格式修复