--- START OF FILE run_npu.sh ---
#!/bin/bash

# ==========================================
# NPU Environment Wrapper
# ==========================================

# 1. 基础 CANN 环境变量 (必需，用于动态库加载)
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 解决 Protobuf 冲突的关键补丁
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 3. 清空 PYTHONPATH，防止系统包干扰虚拟环境
export PYTHONPATH=""

# 4. 激活虚拟环境 (修改为你实际的 venv 路径)
# 假设你的 venv 在 /mnt/xc/mindspore-test/venv
VENV_PATH="/mnt/xc/mindspore-test/venv"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# 5. 启动 Python 负载逻辑
# "$@" 将 C++ 传来的参数 (--profile x --level y) 透传给 Python
exec python3 npu_load_worker.py "$@"