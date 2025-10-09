#!/bin/bash
# 自动安装BitBLAS (修复libtinfo.so.5问题)
# 使用方案1: 禁用LLVM (最简单可靠)

set -e

cd /home/jz97/VLM_REPO/openpi

echo "========================================"
echo "自动安装BitBLAS (W4A8 DuQuant)"
echo "========================================"
echo ""

# 1. 克隆BitBLAS
if [ ! -d "third_party/BitBLAS" ]; then
    echo "步骤1: 克隆BitBLAS仓库..."
    git clone https://github.com/microsoft/BitBLAS.git third_party/BitBLAS
    cd third_party/BitBLAS
    git submodule update --init --recursive
    cd ../..
else
    echo "步骤1: BitBLAS已存在，跳过克隆"
fi

cd third_party/BitBLAS

# 2. 清理之前的build
echo ""
echo "步骤2: 清理之前的编译..."
rm -rf 3rdparty/tvm/build
mkdir -p 3rdparty/tvm/build

# 3. 配置TVM (禁用LLVM以避免libtinfo.so.5问题)
echo ""
echo "步骤3: 配置TVM (runtime-only, 无LLVM)..."
cd 3rdparty/tvm/build

cat > config.cmake <<'EOF'
# Minimal TVM runtime configuration
# 关键: 禁用LLVM以避免libtinfo.so.5依赖问题

set(CMAKE_BUILD_TYPE Release)

# CUDA支持 (必需)
set(USE_CUDA ON)
set(USE_CUBLAS ON)
set(USE_CUTLASS OFF)  # 禁用CUTLASS避免__hfma2编译错误
set(USE_CUDA_PATH /usr/local/cuda)

# 禁用LLVM (避免libtinfo.so.5问题)
set(USE_LLVM OFF)

# 其他必需选项
set(HIDE_PRIVATE_SYMBOLS ON)
set(USE_RPC ON)
set(USE_GRAPH_EXECUTOR ON)
set(USE_PROFILER ON)

# 忽略编译警告
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-uninitialized -Wno-maybe-uninitialized")
EOF

echo "配置文件已创建"

# 4. 编译TVM
echo ""
echo "步骤4: 编译TVM (这需要10-20分钟)..."
echo "使用 $(nproc) 个CPU核心并行编译..."

cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tee cmake.log
echo ""
echo "CMake配置完成，开始编译..."

make -j$(nproc) 2>&1 | tee build.log

# 检查编译结果
if [ -f libtvm.so ] || [ -f libtvm_runtime.so ]; then
    echo "✅ TVM编译成功!"
    ls -lh libtvm*.so
else
    echo "❌ TVM编译失败，检查 build.log"
    exit 1
fi

# 5. 安装BitBLAS Python包
echo ""
echo "步骤5: 安装BitBLAS Python包..."
cd ../../..

pip install -e . --no-build-isolation 2>&1 | tee install.log

# 6. 测试安装
echo ""
echo "========================================"
echo "测试BitBLAS"
echo "========================================"

python -c "
import sys
try:
    import bitblas
    print('✅ BitBLAS安装成功!')
    print(f'   版本: {bitblas.__version__}')
    print(f'   路径: {bitblas.__file__}')

    # 测试linear_w4a8函数
    if hasattr(bitblas, 'linear_w4a8'):
        print('✅ linear_w4a8 kernel可用')
    elif hasattr(bitblas, 'ops') and hasattr(bitblas.ops, 'linear_w4a8'):
        print('✅ bitblas.ops.linear_w4a8 kernel可用')
    else:
        print('⚠️  警告: linear_w4a8 kernel未找到')

    sys.exit(0)

except Exception as e:
    print(f'❌ BitBLAS测试失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ BitBLAS安装完成!"
    echo "========================================"
    echo ""
    echo "现在可以测试真正的W4A8 DuQuant加速:"
    echo ""
    echo "  cd /home/jz97/VLM_REPO/openpi"
    echo "  ./test_llm_duquant.sh"
    echo ""
    echo "预期结果:"
    echo "  - OFF (FP16):  ~11 ms/call"
    echo "  - BitBLAS:     ~2-3 ms/call (3-5x加速!)"
    echo ""
else
    echo ""
    echo "❌ 安装失败，请检查错误信息"
    exit 1
fi
