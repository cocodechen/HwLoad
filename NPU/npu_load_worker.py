import argparse
import time
import signal
import sys
import random
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor, context, Parameter, ParameterTuple
try:
    from npu_train import run_simulation_loop
except ImportError:
    print("Error: 'npu_train.py' not found. Please make sure it exists.")
    sys.exit(1)

# =========================
# 信号处理
# =========================
def signal_handler(sig, frame):
    print("Received SIGTERM, exiting...")
    sys.exit(0)

# =========================
# 配置管理
# =========================
def get_config(profile, level):
    # === 1. Random 模式 (波动负载) ===
    if profile == 'random':
        # 去除 Level 的影响，统一为固定参数
        # N=1024 保证计算强度，Steps=15 保证单次执行时间适中
        return 1024, 15, 0.0

    # === 2. Memory 模式 (显存带宽) ===
    if profile == 'memory':
        # [暴力升级] 之前的 2048 太小，不足以拉满带宽
        if level == 'saturated':
            return 8192, 20, 0.0 # 单块 256MB，极致大块读写
        elif level == 'high':
            return 4096, 40, 0.0 # 单块 64MB
        else:
            return 2048, 50, 0.05

    # === 3. Compute 模式 (算力) ===
    if profile == 'compute':
        if level == 'saturated':
            return 8192, 100, 0.0 # 巨大的矩阵乘法
        base_n = 4096
        base_steps = 200 
    else: # data / default
        base_n = 2048
        base_steps = 50

    # === 通用 Level 调节 (仅对非 Random 生效) ===
    if level == 'idle':
        return 64, 1, 1.0
    elif level == 'low':
        return base_n, int(base_steps * 0.2), 0.05
    elif level == 'medium':
        return base_n, int(base_steps * 0.5), 0.01
    elif level == 'high':
        return int(base_n * 1.0), base_steps, 0.0
    elif level == 'saturated':
        return int(base_n * 1.5), int(base_steps * 1.5), 0.0
    
    return 1024, 10, 0.1

# =========================
# 负载基类
# =========================
class LoadBase(nn.Cell):
    def __init__(self, N, steps, dtype):
        super().__init__()
        self.steps = steps
        self.dtype = dtype
        self.reduce = ops.ReduceSum()

    def finish_op(self, x):
        return self.reduce(x)

# =========================
# 1. Compute Load (Cube 密集)
# =========================
class ComputeCell(LoadBase):
    def __init__(self, N, steps):
        super().__init__(N, steps, ms.float16)
        self.matmul = ops.MatMul()
        self.x = Parameter(Tensor(np.random.normal(0, 0.01, (N, N)), self.dtype), name="x")
        self.y = Parameter(Tensor(np.random.normal(0, 0.01, (N, N)), self.dtype), name="y")

    def construct(self, _=None):
        out = self.x
        for _ in range(self.steps):
            out = self.matmul(out, self.y)
        return self.finish_op(out)

# =========================
# 2. Memory Load (HBM 带宽密集)
# =========================
class MemoryCell(LoadBase):
    def __init__(self, N, steps):
        super().__init__(N, steps, ms.float32)
        self.add = ops.Add()
        self.assign = ops.Assign()
        self.depend = ops.Depend()
        
        # [策略调整] 
        # 如果 N 很大(8192, 256MB)，Buffer 少一点 (8个=2GB) 防止 OOM
        # 如果 N 较小(2048, 16MB)，Buffer 多一点 (32个=512MB) 击穿 Cache
        if N >= 8192:
            self.buffer_count = 8
        elif N >= 4096:
            self.buffer_count = 16
        else:
            self.buffer_count = 32
            
        print(f"Memory Load: Allocating {self.buffer_count} blocks of {N}x{N} FP32 (Total: ~{self.buffer_count * N * N * 4 / 1024 / 1024:.2f} MB)")
        
        self.buffers = ParameterTuple([
            Parameter(Tensor(np.ones((N, N)), self.dtype), name=f"buf_{i}")
            for i in range(self.buffer_count)
        ])
        self.val = Tensor(0.0001, self.dtype)

    def construct(self, _=None):
        last_op = self.val 
        # 为了让流水线更紧凑，不再 Unroll 太多，而是依赖算子本身的大数据量
        for i in range(self.buffer_count):
            buf = self.buffers[i]
            current_buf = self.depend(buf, last_op)
            res = self.add(current_buf, self.val)
            assign_op = self.assign(buf, res)
            last_op = assign_op
        return self.finish_op(self.buffers[0])

# =========================
# 3. Data Load
# =========================
class DataCell(LoadBase):
    def __init__(self, N, steps):
        super().__init__(N, steps, ms.float32)
        self.matmul = ops.MatMul()
        self.cast = ops.Cast()
        self.src = Parameter(Tensor(np.random.normal(0, 1, (N, N)), ms.float32), name="src")
        self.weight = Parameter(Tensor(np.random.normal(0, 1, (N, N)), ms.float16), name="w")

    def construct(self, _=None):
        x = self.src
        w = self.weight
        for _ in range(self.steps):
            x_16 = self.cast(x, ms.float16)
            res = self.matmul(x_16, w)
            x = self.cast(res, ms.float32)
        return self.finish_op(x)


def run_loop(net, sleep_interval, profile, N):
    print("Warmup (Compiling graph)...")
    try:
        # 不需要传参数，或者传个 None 占位
        net(None)
    except Exception as e:
        print(f"Error during warmup: {e}")
        sys.exit(1)
    
    print(f"Load started. Profile: {profile} (Running FULL SPEED)")
    
    while True:
        # === [优化] 2. 移除 CPU 端的数据生成 ===
        # === [优化] 3. 移除 .asnumpy() 同步 ===
        # 仅仅下发任务，不做结果回传，让 NPU 跑死
        net(None)
        
        # 仅在需要波动的模式下加入 sleep，否则全速运行
        if profile != 'random' and sleep_interval > 0:
            time.sleep(sleep_interval)

def main():
    parser = argparse.ArgumentParser()
    # 增加提示：推荐使用 random 模式
    parser.add_argument('--profile', type=str, default='random', 
                        help='random (REAL TRAINING SIMULATION), compute, memory')
    parser.add_argument('--level', type=str, default='high', help='(Legacy) idle/low/medium/high')
    args = parser.parse_args()

    try:
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    except Exception as e:
        print(f"Init Context Failed: {e}")
        sys.exit(1)

    if args.profile == 'random':
        # 调用新的模拟器，忽略 level 参数，使用 SIM_CONFIG
        run_simulation_loop()
    else:
        N, steps, sleep_interval = get_config(args.profile, args.level)
        print(f"Config: Profile={args.profile}, Level={args.level} -> N={N}, Steps={steps}")

        if args.profile == 'compute':
            net = ComputeCell(N, steps)
        elif args.profile == 'memory':
            net = MemoryCell(N, steps)
        elif args.profile == 'data':
            net = DataCell(N, steps)
        else:
            net = ComputeCell(N, steps)
        net.set_train(False)
        run_loop(net, sleep_interval, args.profile, N)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    main()