#!/bin/bash
# 使用 UV 创建 OpenPI 新虚拟环境的一键安装脚本

set -e

VENV_NAME=${VENV_NAME:-.venv_test}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}

echo "=========================================="
echo "OpenPI UV 环境 Setup"
echo "虚拟环境: $VENV_NAME"
echo "Python: $PYTHON_VERSION"
echo "=========================================="
echo ""

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ uv 未安装"
    echo ""
    echo "请先安装 uv:"
    echo "  pip install uv"
    echo "或"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv 已安装: $(uv --version)"
echo ""

# 导航到 OpenPI 仓库
cd ~/VLM_REPO/openpi || {
    echo "❌ OpenPI 仓库不存在: ~/VLM_REPO/openpi"
    exit 1
}

echo "✓ 当前目录: $(pwd)"
echo ""

# 检查环境是否已存在
if [ -d "$VENV_NAME" ]; then
    echo "⚠️  虚拟环境 '$VENV_NAME' 已存在"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除现有环境..."
        rm -rf "$VENV_NAME"
    else
        echo "已取消。请使用不同的 VENV_NAME"
        exit 1
    fi
fi

# Step 1: 创建虚拟环境
echo ""
echo "[1/7] 创建 UV 虚拟环境..."
uv venv "$VENV_NAME" --python "$PYTHON_VERSION"
echo "✓ 虚拟环境已创建: $VENV_NAME"

# 激活环境
echo ""
echo "[2/7] 激活环境..."
source "$VENV_NAME/bin/activate"

if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ 环境激活失败"
    exit 1
fi
echo "✓ 环境已激活: $VIRTUAL_ENV"

# Step 2: 安装 OpenPI
echo ""
echo "[3/7] 安装 OpenPI 及所有依赖（这可能需要5-10分钟）..."
uv pip install -e . --no-cache
echo "✓ OpenPI 已安装"

# Step 3: 安装 robosuite 和 dm-control（LIBERO 依赖）
echo ""
echo "[4/7] 安装 LIBERO 依赖（robosuite, dm-control, pyyaml）..."
uv pip install robosuite dm-control pyyaml --no-cache
echo "✓ Robosuite, DM-Control 和 PyYAML 已安装"

# Step 4: 安装 LIBERO
echo ""
echo "[5/7] 安装 LIBERO..."
if [ -d "third_party/libero" ]; then
    cd third_party/libero
    uv pip install -e . --no-cache
    cd ../..
    echo "✓ LIBERO 已安装"
else
    echo "⚠️  LIBERO 未找到: third_party/libero"
fi

# Step 5: 安装 BitBLAS（可选）
echo ""
echo "[6/7] 安装 BitBLAS（可选）..."
if [ -d "third_party/BitBLAS" ]; then
    cd third_party/BitBLAS
    uv pip install -e . --no-cache || echo "⚠️  BitBLAS 安装失败（可选，继续）"
    cd ../..
    echo "✓ BitBLAS 已安装"
else
    echo "⚠️  BitBLAS 未找到（跳过，可选）"
fi

# Step 6: 验证安装
echo ""
echo "[7/7] 验证安装..."
python << 'EOF'
import sys
import torch
import jax

print("\n" + "="*60)
print("安装验证")
print("="*60)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
print(f"JAX: {jax.__version__}")

try:
    import libero
    print("LIBERO: ✓")
except:
    print("LIBERO: ✗ 未安装")
    sys.exit(1)

try:
    import openpi
    print("OpenPI: ✓")
except:
    print("OpenPI: ✗ 未安装")
    sys.exit(1)

try:
    from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()['libero_10']()
    print(f"LIBERO Benchmark: {bench.n_tasks} tasks")
except Exception as e:
    print(f"LIBERO Benchmark: ✗ {e}")

print("="*60)

if not torch.cuda.is_available():
    print("\n⚠️  CUDA 不可用！请检查 CUDA 安装")
    sys.exit(1)

print("\n✓ 所有核心包验证成功！")
print("="*60)
EOF

VERIFY_EXIT=$?
if [ $VERIFY_EXIT -ne 0 ]; then
    echo ""
    echo "❌ 验证失败，请检查错误信息"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup 完成！"
echo "=========================================="
echo ""
echo "虚拟环境: $VENV_NAME"
echo ""
echo "使用方法:"
echo "  1. 激活环境:"
echo "     source $VENV_NAME/bin/activate"
echo ""
echo "  2. 设置 PYTHONPATH:"
echo "     cd ~/VLM_REPO/openpi"
echo "     export PYTHONPATH=\$PWD/src:\$PWD/third_party/libero"
echo ""
echo "  3. 设置 checkpoint:"
echo "     export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch"
echo ""
echo "  4. 运行脚本:"
echo "     ./examples/libero/run_quantvla.sh"
echo ""
echo "快速测试:"
echo "  ./examples/libero/run_fp_baseline.sh"
echo ""
echo "=========================================="
echo ""

# 保存激活命令到文件
cat > activate_env.sh << EOFACTIVATE
#!/bin/bash
# 快速激活脚本
source $VENV_NAME/bin/activate
cd ~/VLM_REPO/openpi
export PYTHONPATH=\$PWD/src:\$PWD/third_party/libero
export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch
echo "✓ 环境已激活并配置"
echo "当前目录: \$(pwd)"
echo "PYTHONPATH: \$PYTHONPATH"
echo "CKPT: \$CKPT"
EOFACTIVATE

chmod +x activate_env.sh

echo "💡 Tip: 下次可以直接运行:"
echo "   source activate_env.sh"
echo ""

deactivate

echo "Setup 脚本完成！"
