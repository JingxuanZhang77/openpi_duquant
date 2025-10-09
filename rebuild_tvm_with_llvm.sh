#!/bin/bash
set -e

echo "重新编译TVM（启用LLVM支持）"
echo "================================"

cd /home/jz97/VLM_REPO/openpi/third_party/BitBLAS/3rdparty/tvm

# 1. 清理之前的编译
echo "步骤1: 清理之前的编译..."
rm -rf build
mkdir -p build
cd build

# 2. 配置TVM - 启用LLVM
echo "步骤2: 配置TVM（启用LLVM）..."
cat > config.cmake <<EOF
# Licensed to the Apache Software Foundation (ASF)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

# CUDA支持 (必需)
set(USE_CUDA ON)
set(USE_CUBLAS ON)
set(USE_CUTLASS OFF)  # 禁用CUTLASS避免__hfma2编译错误
set(USE_CUDA_PATH /usr/local/cuda)

# 启用LLVM (BitBLAS INT4 kernel需要)
set(USE_LLVM "/usr/lib/llvm-14/bin/llvm-config")  # 使用系统LLVM

# Runtime only (不需要完整的编译器)
set(BUILD_STATIC_RUNTIME ON)
set(SUMMARIZE OFF)
set(USE_RANDOM ON)

# 禁用不需要的功能
set(USE_OPENCL OFF)
set(USE_METAL OFF)
set(USE_VULKAN OFF)
set(USE_ROCM OFF)

# 线程支持
set(USE_OPENMP gnu)
set(USE_PTHREADS ON)

# 其他设置
set(INSTALL_DEV ON)
set(HIDE_PRIVATE_SYMBOLS ON)
EOF

echo "配置内容:"
cat config.cmake

# 3. CMake配置
echo "步骤3: 运行CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# 4. 编译TVM
echo "步骤4: 编译TVM（使用8核）..."
make -j8

# 5. 测试LLVM是否可用
echo "步骤5: 测试LLVM支持..."
python3 -c "import sys; sys.path.insert(0, '$(pwd)/../python'); import tvm; print('TVM version:', tvm.__version__); print('LLVM enabled:', 'llvm' in tvm.runtime.enabled())"

echo ""
echo "✅ TVM重新编译完成（已启用LLVM）"
echo "现在可以测试BitBLAS INT4内核了"
