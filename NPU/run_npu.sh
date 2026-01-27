#!/bin/bash

# ==========================================
# 1. 自动定位脚本和环境路径
# ==========================================
# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 假设 .py 文件和 .sh 在同一目录
PY_SCRIPT_PATH="${SCRIPT_DIR}/npu_load_worker.py"

# 假设 venv 在 NPU 目录的上一级
# 如果你的位置不同，请修改这里
VENV_PATH="${SCRIPT_DIR}/../venv"

# ==========================================
# 2. 基础驱动环境 (CANN) - 必须保留
# ==========================================
# C++ 启动的子shell默认没有环境变量，必须手动source
# 否则 NPU 驱动库 (libascendcl.so) 找不到，程序无法初始化硬件
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ==========================================
# 3. 激活 Python 虚拟环境
# ==========================================
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "[Error] Virtual environment not found at: ${VENV_PATH}"
    exit 1
fi

# ==========================================
# 4. 执行
# ==========================================
if [ ! -f "$PY_SCRIPT_PATH" ]; then
    echo "[Error] Python script not found at: $PY_SCRIPT_PATH"
    exit 1
fi

# 直接透传参数
exec python3 "$PY_SCRIPT_PATH" "$@"