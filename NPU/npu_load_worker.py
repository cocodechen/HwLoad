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

# =========================
# 4. Random Load (MobileNet Block)
# =========================
class RandomCell(LoadBase):
    def __init__(self, N, steps):
        super().__init__(N, steps, ms.float16)
        in_channel = N
        # 增大 N 或者分辨率以增加计算密度
        # 如果 N=1024 负载仍不满，可以加大到 2048
        
        # === [优化] 1. 输入直接定义在 Device 上 ===
        # 模拟 batch_size=32, h=32, w=32 (加大分辨率以增加计算密度)
        self.input_data = Parameter(Tensor(np.random.normal(0, 1, (32, N, 32, 32)), ms.float16), name="fixed_input")
        
        expand_ratio = 4 
        hidden_dim = in_channel * expand_ratio
        
        self.conv_pw = nn.Conv2d(in_channel, hidden_dim, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        self.conv_dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, 
                                 group=hidden_dim, pad_mode='same', has_bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.conv_proj = nn.Conv2d(hidden_dim, in_channel, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(in_channel)
        
        self.relu = nn.ReLU()
        self.add = ops.Add()
        
        self.to_float(ms.float16)

    def construct(self, _=None): # 忽略外部输入
        # === [优化] 直接使用内部 Parameter ===
        out = self.input_data 
        
        for _ in range(self.steps):
            identity = out
            out = self.conv_pw(out)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv_dw(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv_proj(out)
            out = self.bn3(out)
            out = self.add(out, identity)
        return self.finish_op(out)

# =========================
# 辅助函数
# =========================
def create_random_input(profile, N):
    if profile != 'random':
        return None
    # FP16 Input
    return Tensor(np.random.randn(32, N, 14, 14).astype(np.float16))

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
    parser.add_argument('--profile', type=str, default='compute', help='compute/memory/data/random')
    parser.add_argument('--level', type=str, default='low', help='idle/low/medium/high/saturated')
    args = parser.parse_args()

    try:
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    except Exception as e:
        print(f"Init Context Failed: {e}")
        sys.exit(1)

    N, steps, sleep_interval = get_config(args.profile, args.level)
    
    # [修改点] UI 显示优化
    display_level = "Fluctuation" if args.profile == 'random' else args.level
    print(f"Config: Profile={args.profile}, Level={display_level} -> N={N}, Steps={steps}")

    if args.profile == 'compute':
        net = ComputeCell(N, steps)
    elif args.profile == 'memory':
        net = MemoryCell(N, steps)
    elif args.profile == 'random':
        net = RandomCell(N, steps)
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