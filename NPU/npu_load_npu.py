import argparse
import time
import signal
import sys
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor, context

# 优雅处理 SIGTERM，方便 C++ stop() 停止
def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

def get_params(level):
    """
    根据 Level 定义矩阵大小 (N x N)
    Ascend 910 算力很强，矩阵要够大才能打满
    """
    if level == 'low':       return 2048
    elif level == 'medium':  return 4096
    elif level == 'high':    return 8192
    elif level == 'saturated': return 16384 # 极大矩阵，需要较大显存
    return 1024

def run_compute(N):
    """
    计算密集型: MatMul (矩阵乘法)
    主要利用 Cube Unit
    """
    x = Tensor(np.ones([N, N]), ms.float16) # FP16 是 NPU 计算最快的格式
    y = Tensor(np.ones([N, N]), ms.float16)
    matmul = ops.MatMul()
    
    print(f"Running COMPUTE load. Matrix: {N}x{N}, FP16")
    while True:
        _ = matmul(x, y)

def run_memory(N):
    """
    访存密集型: Element-wise Ops + Broadcast
    主要利用 Vector Unit 和 HBM 带宽
    """
    # 稍微调大一点 N，因为 Vector 操作比 Cube 快
    N = int(N * 1.5) 
    x = Tensor(np.ones([N, N]), ms.float32) # FP32 占用显存带宽更大
    add = ops.Add()
    mul = ops.Mul()
    
    print(f"Running MEMORY load. Tensor: {N}x{N}, FP32")
    while True:
        res = add(x, 1.0)
        _ = mul(res, 2.0)

def run_data(N):
    """
    Data/混合型: 模拟复杂数据处理
    混合 MatMul 和 Cast 操作
    """
    x = Tensor(np.random.normal(0, 1, (N, N)), ms.float32)
    y = Tensor(np.random.normal(0, 1, (N, N)), ms.float32)
    
    matmul = ops.MatMul()
    cast = ops.Cast()
    
    print(f"Running DATA load. Matrix: {N}x{N}, FP32<->FP16 Mix")
    while True:
        # 强制类型转换消耗 Vector 资源
        x_16 = cast(x, ms.float16)
        y_16 = cast(y, ms.float16)
        # 计算
        res = matmul(x_16, y_16)
        # 转回
        _ = cast(res, ms.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=str, default='compute', help='compute/memory/data')
    parser.add_argument('--level', type=str, default='low', help='low/medium/high/saturated')
    args = parser.parse_args()

    # 初始化 NPU 环境
    try:
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        # 随便创建一个 Tensor 触发 lazy initialization
        _ = Tensor(np.array([1.0]), ms.float32)
    except Exception as e:
        print(f"Failed to init MindSpore: {e}")
        sys.exit(1)

    N = get_params(args.level)
    
    # 根据 profile 执行死循环
    if args.profile == 'compute':
        run_compute(N)
    elif args.profile == 'memory':
        run_memory(N)
    elif args.profile == 'data':
        run_data(N)
    else:
        run_compute(N)

if __name__ == "__main__":
    main()