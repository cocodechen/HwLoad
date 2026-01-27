# chmod +x setup_mindspore.sh
# ./set_env.sh 2.3.1

#!/usr/bin/env bash
set -e

########################
# 可配置参数
########################
MS_VERSION=${1:-2.3.1}      # MindSpore 版本，默认 2.3.1
VENV_NAME=venv

########################
# 基础检查
########################
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found"
    exit 1
fi

PY_VER=$(python3 - <<EOF
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
EOF
)

case "$PY_VER" in
  3.9|3.10|3.11)
    ;;
  *)
    echo "[ERROR] Python version must be >=3.9, current: $PY_VER"
    exit 1
    ;;
esac

PY_TAG="cp${PY_VER/./}"

ARCH=$(uname -m)
case "$ARCH" in
  aarch64)
    MS_ARCH="aarch64"
    ;;
  x86_64)
    MS_ARCH="x86_64"
    ;;
  *)
    echo "[ERROR] Unsupported arch: $ARCH"
    exit 1
    ;;
esac

echo "[INFO] Python: $PY_VER ($PY_TAG)"
echo "[INFO] Arch:   $MS_ARCH"
echo "[INFO] MS Ver: $MS_VERSION"

########################
# 创建虚拟环境
########################
python3 -m venv ${VENV_NAME}
source ${VENV_NAME}/bin/activate

pip install --upgrade pip

########################
# 基础依赖
########################
pip install sympy
pip install "numpy>=1.20.0,<2.0.0"

########################
# Ascend 依赖
########################
ASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest/lib64

pip install ${ASCEND_ROOT}/te-*-py3-none-any.whl
pip install ${ASCEND_ROOT}/hccl-*-py3-none-any.whl

########################
# 安装 MindSpore
########################
MS_WHL="mindspore-${MS_VERSION}-${PY_TAG}-${PY_TAG}-linux_${MS_ARCH}.whl"
MS_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/${MS_ARCH}/${MS_WHL}"

echo "[INFO] Installing MindSpore from:"
echo "       ${MS_URL}"

pip install "${MS_URL}" \
  --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

########################
# 验证
########################
python - <<EOF
import mindspore
mindspore.set_context(device_target='Ascend')
mindspore.run_check()
print("[SUCCESS] MindSpore Ascend is OK")
EOF
