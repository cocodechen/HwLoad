#!/bin/bash

# 定义虚拟环境目录
VENV_DIR="loadgen_venv"

# 检查 Python 版本
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: python3 could not be found."
    exit 1
fi

# 获取 Python 版本 (简单的字符串比较)
PY_VER=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: $PY_VER"

# 简单检查是否在 3.10 - 3.12 之间 (根据需求)
# 注意：这里仅作提示，不强制阻止运行，因为通常 3.8+ 都兼容
if [[ "$PY_VER" != "3.10" && "$PY_VER" != "3.11" && "$PY_VER" != "3.12" ]]; then
    echo "Warning: Recommended Python version is 3.10 to 3.12. Current is $PY_VER."
fi

# 1. 创建虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in ./$VENV_DIR ..."
    $PYTHON_CMD -m venv $VENV_DIR
else
    echo "Virtual environment already exists."
fi

# 2. 激活环境
source $VENV_DIR/bin/activate

# 3. 升级 pip
pip install --upgrade pip

# 4. 安装依赖
# numpy: 矩阵计算
# psutil: 内存和系统监控
# torch: GPU 计算 (这里默认安装 CPU+CUDA 的通用版本，具体根据机器环境可能需要指定 index-url)
echo "Installing dependencies..."
pip install numpy psutil torch

echo "---------------------------------------------------"
echo "Setup complete."
echo "To run the load generator:"
echo "  source $VENV_DIR/bin/activate"
echo "  python main.py cpu:random gpu:random"
echo "---------------------------------------------------"