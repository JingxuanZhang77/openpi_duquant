#!/bin/bash
# 修复BitBLAS编译问题并安装
# 问题: 缺少 libtinfo.so.5
# 解决: 使用系统LLVM或禁用LLVM

set -e

cd /home/jz97/VLM_REPO/openpi

echo "========================================"
echo "安装BitBLAS (修复版)"
echo "========================================"

# 1. 克隆BitBLAS
if [ ! -d "third_party/BitBLAS" ]; then
    echo "克隆BitBLAS..."
    git clone https://github.com/microsoft/BitBLAS.git third_party/BitBLAS
    cd third_party/BitBLAS
    git submodule update --init --recursive
    cd ../..
fi

cd third_party/BitBLAS

# 2. 修复libtinfo.so.5问题的3个方案

echo ""
echo "选择修复方案:"
echo "1. 禁用LLVM (最简单，推荐)"
echo "2. 使用系统LLVM"
echo "3. 创建libtinfo.so.5软链接"
echo ""
read -p "请选择 (1/2/3, 默认1): " choice
choice=${choice:-1}

case $choice in
    1)
        echo "方案1: 禁用LLVM (runtime-only build)"

        # 配置TVM (不需要LLVM)
        cd 3rdparty/tvm
        mkdir -p build
        cd build

        cat > config.cmake <<'EOF'
# Minimal TVM runtime configuration (no LLVM)
set(CMAKE_BUILD_TYPE Release)
set(USE_CUDA ON)
set(USE_CUBLAS ON)
set(USE_CUTLASS ON)
set(USE_CUDA_PATH /usr/local/cuda)

# 关键: 禁用LLVM
set(USE_LLVM OFF)

# 其他设置
set(HIDE_PRIVATE_SYMBOLS ON)
set(USE_RPC ON)
set(USE_GRAPH_EXECUTOR ON)
set(USE_PROFILER ON)

# 忽略警告
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-uninitialized")
EOF

        echo "配置完成，开始编译TVM..."
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc)

        cd ../../..
        ;;

    2)
        echo "方案2: 使用系统LLVM"

        # 检查系统LLVM
        if ! command -v llvm-config &> /dev/null; then
            echo "安装系统LLVM..."
            sudo apt-get update
            sudo apt-get install -y llvm-14 llvm-14-dev
        fi

        LLVM_CONFIG=$(which llvm-config || echo "/usr/bin/llvm-config-14")
        echo "使用LLVM: $LLVM_CONFIG"

        cd 3rdparty/tvm
        mkdir -p build
        cd build

        cat > config.cmake <<EOF
set(CMAKE_BUILD_TYPE Release)
set(USE_CUDA ON)
set(USE_CUBLAS ON)
set(USE_LLVM $LLVM_CONFIG)
set(HIDE_PRIVATE_SYMBOLS ON)
set(CMAKE_CXX_FLAGS "\${CMAKE_CXX_FLAGS} -Wno-uninitialized")
EOF

        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc)

        cd ../../..
        ;;

    3)
        echo "方案3: 创建libtinfo.so.5软链接"

        # 找到libtinfo.so.6
        LIBTINFO6=$(find /lib /usr/lib -name "libtinfo.so.6*" 2>/dev/null | head -1)

        if [ -z "$LIBTINFO6" ]; then
            echo "错误: 找不到libtinfo.so.6"
            exit 1
        fi

        echo "找到: $LIBTINFO6"

        # 创建软链接到conda环境
        CONDA_LIB="$CONDA_PREFIX/lib"
        if [ -d "$CONDA_LIB" ]; then
            ln -sf "$LIBTINFO6" "$CONDA_LIB/libtinfo.so.5"
            echo "创建软链接: $CONDA_LIB/libtinfo.so.5 -> $LIBTINFO6"
        fi

        # 继续使用预下载的LLVM
        cd 3rdparty/tvm
        mkdir -p build
        cd build

        cp ../cmake/config.cmake .
        sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/' config.cmake
        sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/' config.cmake

        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc)

        cd ../../..
        ;;
esac

# 3. 安装BitBLAS Python包
echo ""
echo "安装BitBLAS Python包..."
cd /home/jz97/VLM_REPO/openpi/third_party/BitBLAS
pip install -e . --no-build-isolation

# 4. 测试
echo ""
echo "========================================"
echo "测试BitBLAS安装"
echo "========================================"
python -c "import bitblas; print('✅ BitBLAS version:', bitblas.__version__)" || echo "❌ BitBLAS导入失败"

echo ""
echo "========================================"
echo "安装完成!"
echo "========================================"
echo ""
echo "现在可以测试真正的W4A8 DuQuant:"
echo "  cd /home/jz97/VLM_REPO/openpi"
echo "  ./test_llm_duquant.sh"
