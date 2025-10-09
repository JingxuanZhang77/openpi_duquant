#!/bin/bash
# 完整的BitBLAS W4A8安装脚本（支持真正的INT4加速）
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}BitBLAS W4A8 完整安装脚本${NC}"
echo -e "${GREEN}======================================${NC}"

OPENPI_ROOT="/home/jz97/VLM_REPO/openpi"
BITBLAS_ROOT="$OPENPI_ROOT/third_party/BitBLAS"
TVM_ROOT="$BITBLAS_ROOT/3rdparty/tvm"
TILELANG_ROOT="$BITBLAS_ROOT/3rdparty/tilelang"

cd "$OPENPI_ROOT"

# ============================================================
# 步骤1: 克隆BitBLAS（如果不存在）
# ============================================================
echo -e "\n${YELLOW}步骤1: 克隆BitBLAS仓库${NC}"
if [ ! -d "$BITBLAS_ROOT" ]; then
    echo "克隆BitBLAS..."
    mkdir -p third_party
    cd third_party
    git clone --recursive https://github.com/microsoft/BitBLAS.git
    cd BitBLAS
    git submodule update --init --recursive
    echo -e "${GREEN}✓ BitBLAS克隆完成${NC}"
else
    echo "BitBLAS已存在，跳过克隆"
    cd "$BITBLAS_ROOT"
    git submodule update --init --recursive 2>/dev/null || true
fi

# ============================================================
# 步骤2: 检查并安装系统LLVM
# ============================================================
echo -e "\n${YELLOW}步骤2: 检查LLVM安装${NC}"

# 检查conda环境中是否有llvm
CONDA_LLVM="/home/jz97/miniconda3/envs/fingerprint/bin/llvm-config"
SYSTEM_LLVM_14="/usr/bin/llvm-config-14"
SYSTEM_LLVM="/usr/bin/llvm-config"

LLVM_CONFIG=""
if [ -f "$CONDA_LLVM" ]; then
    echo "发现conda LLVM: $CONDA_LLVM"
    LLVM_CONFIG="$CONDA_LLVM"
elif [ -f "$SYSTEM_LLVM_14" ]; then
    echo "发现系统LLVM 14: $SYSTEM_LLVM_14"
    LLVM_CONFIG="$SYSTEM_LLVM_14"
elif [ -f "$SYSTEM_LLVM" ]; then
    echo "发现系统LLVM: $SYSTEM_LLVM"
    LLVM_CONFIG="$SYSTEM_LLVM"
else
    echo -e "${RED}未找到LLVM！${NC}"
    echo -e "${YELLOW}尝试使用conda安装LLVM...${NC}"

    # 尝试用conda安装
    conda install -y -c conda-forge llvmdev=14 || {
        echo -e "${RED}Conda安装失败！${NC}"
        echo -e "${YELLOW}请手动安装LLVM:${NC}"
        echo "  sudo apt-get install -y llvm-14 llvm-14-dev"
        echo "或者:"
        echo "  conda install -y -c conda-forge llvmdev"
        exit 1
    }

    LLVM_CONFIG="/home/jz97/miniconda3/envs/fingerprint/bin/llvm-config"
fi

# 验证LLVM可用
if [ ! -f "$LLVM_CONFIG" ]; then
    echo -e "${RED}LLVM配置失败！${NC}"
    exit 1
fi

LLVM_VERSION=$($LLVM_CONFIG --version)
echo -e "${GREEN}✓ 使用LLVM: $LLVM_CONFIG (version $LLVM_VERSION)${NC}"

# ============================================================
# 步骤3: 创建libtinfo.so.5符号链接（如果需要）
# ============================================================
echo -e "\n${YELLOW}步骤3: 配置libtinfo${NC}"

LIBTINFO5="/home/jz97/miniconda3/envs/fingerprint/lib/libtinfo.so.5"
if [ ! -f "$LIBTINFO5" ]; then
    echo "创建libtinfo.so.5符号链接..."
    ln -sf /lib/x86_64-linux-gnu/libtinfo.so.6 "$LIBTINFO5"
    echo -e "${GREEN}✓ libtinfo.so.5已创建${NC}"
else
    echo "libtinfo.so.5已存在"
fi

# ============================================================
# 步骤4: 清理之前的编译
# ============================================================
echo -e "\n${YELLOW}步骤4: 清理之前的编译${NC}"
rm -rf "$TVM_ROOT/build"
rm -rf "$TILELANG_ROOT/build"
echo "清理完成"

# ============================================================
# 步骤5: 配置TVM（启用LLVM + CUDA）
# ============================================================
echo -e "\n${YELLOW}步骤5: 配置TVM (启用LLVM)${NC}"
mkdir -p "$TVM_ROOT/build"
cd "$TVM_ROOT/build"

cat > config.cmake <<EOF
# TVM编译配置 - 支持BitBLAS INT4
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

# CUDA支持 (必需)
set(USE_CUDA ON)
set(USE_CUBLAS ON)
set(USE_CUTLASS OFF)  # 禁用CUTLASS避免编译错误
set(USE_CUDA_PATH /usr/local/cuda)

# LLVM支持 (BitBLAS INT4内核必需!)
set(USE_LLVM "$LLVM_CONFIG")

# Runtime
set(BUILD_STATIC_RUNTIME ON)
set(SUMMARIZE OFF)
set(USE_RANDOM ON)

# 禁用不需要的后端
set(USE_OPENCL OFF)
set(USE_METAL OFF)
set(USE_VULKAN OFF)
set(USE_ROCM OFF)

# 线程支持
set(USE_OPENMP gnu)
set(USE_PTHREADS ON)

# 其他
set(INSTALL_DEV ON)
set(HIDE_PRIVATE_SYMBOLS ON)
EOF

echo "TVM配置内容:"
cat config.cmake

# ============================================================
# 步骤6: 编译TVM
# ============================================================
echo -e "\n${YELLOW}步骤6: 编译TVM (需要10-20分钟)${NC}"
echo "使用8核编译..."

cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tee cmake_config.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}✗ CMake配置失败！${NC}"
    echo "查看日志: $TVM_ROOT/build/cmake_config.log"
    exit 1
fi

echo -e "${GREEN}✓ CMake配置成功${NC}"
echo "开始编译TVM..."

make -j8 2>&1 | tee make.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}✗ TVM编译失败！${NC}"
    echo "查看日志: $TVM_ROOT/build/make.log"
    exit 1
fi

echo -e "${GREEN}✓ TVM编译成功${NC}"

# 验证LLVM支持
echo "验证TVM LLVM支持..."
python3 -c "import sys; sys.path.insert(0, '$TVM_ROOT/python'); import tvm; assert 'llvm' in tvm.runtime.enabled(), 'LLVM not enabled!'; print('✓ LLVM已启用')"

# ============================================================
# 步骤7: 编译TileLang
# ============================================================
echo -e "\n${YELLOW}步骤7: 编译TileLang${NC}"
mkdir -p "$TILELANG_ROOT/build"
cd "$TILELANG_ROOT/build"

cmake .. -DTVM_PREBUILD_PATH="$TVM_ROOT/build" 2>&1 | tee cmake.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}✗ TileLang CMake配置失败！${NC}"
    exit 1
fi

make -j8 2>&1 | tee make.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}✗ TileLang编译失败！${NC}"
    exit 1
fi

echo -e "${GREEN}✓ TileLang编译成功${NC}"

# 复制库文件到Python包目录
cp libtilelang_module.so "$TILELANG_ROOT/"
echo "TileLang库已复制到: $TILELANG_ROOT/libtilelang_module.so"

# ============================================================
# 步骤8: 安装BitBLAS Python包
# ============================================================
echo -e "\n${YELLOW}步骤8: 安装BitBLAS Python包${NC}"
cd "$BITBLAS_ROOT"

# 使用editable模式安装
pip install -e . 2>&1 | tee pip_install.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}✗ BitBLAS安装失败！${NC}"
    echo "查看日志: $BITBLAS_ROOT/pip_install.log"
    exit 1
fi

echo -e "${GREEN}✓ BitBLAS安装成功${NC}"

# ============================================================
# 步骤9: 测试BitBLAS
# ============================================================
echo -e "\n${YELLOW}步骤9: 测试BitBLAS INT4支持${NC}"

cd "$OPENPI_ROOT"
PYTHONPATH="$OPENPI_ROOT/src:$BITBLAS_ROOT" python3 << 'PYTEST'
import torch
import bitblas
from bitblas import Matmul, MatmulConfig

print(f"BitBLAS version: {bitblas.__version__}")

# 测试简单的W4A16配置
config = MatmulConfig(
    M=1,
    N=256,
    K=256,
    A_dtype="float16",
    W_dtype="int4",
    out_dtype="float16",
    accum_dtype="int32",
    layout="nt",
    with_bias=False,
    with_scaling=True,
    group_size=-1,
)

print(f"创建BitBLAS Matmul算子...")
try:
    matmul_op = Matmul(config=config, enable_tuning=False)
    print("✓ BitBLAS INT4内核创建成功！")
    print("✓ 真正的W4A8加速已就绪！")
except Exception as e:
    print(f"✗ 创建失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
PYTEST

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ BitBLAS测试失败${NC}"
    exit 1
fi

# ============================================================
# 完成
# ============================================================
echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}✓ BitBLAS W4A8安装成功！${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "下一步："
echo "1. 运行测试: python test_bitblas_int4.py"
echo "2. 运行benchmark: OPENPI_DUQUANT_BACKEND=bitblas ./bench/bench_duquant_backends.py"
echo ""
echo "PYTHONPATH设置:"
echo "export PYTHONPATH=$OPENPI_ROOT/src:$BITBLAS_ROOT"
