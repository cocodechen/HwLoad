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
    # === 基础参数 ===
    if profile == 'compute':
        base_n = 4096
        base_steps = 200 
    elif profile == 'memory':
        base_n = 2048 
        base_steps = 50 
    elif profile == 'random':
        # N 控制通道数，1024 算是中等偏大负载
        base_n = 1024
        # 步数适中，模拟一个深层网络的推理耗时
        base_steps = 15 
    else: 
        base_n = 2048
        base_steps = 50

    # === Level 调节 ===
    if profile == 'random':
        # [修改点1] Random 模式下，Level 决定波动的剧烈程度，不再返回固定 sleep
        # 这里返回的 sleep_interval 将作为一个“基准值”或者“最大值”
        if level == 'low': return base_n, base_steps, 0.5   # 波动稀疏
        if level == 'medium': return base_n, base_steps, 0.1 # 波动频繁
        if level == 'high': return base_n, base_steps, 0.01  # 极其剧烈
        return base_n, base_steps, 0.2
        
    # 其他模式保持固定逻辑
    if level == 'idle':
        return 64, 1, 1.0
    elif level == 'low':
        return base_n, int(base_steps * 0.2), 0.05
    elif level == 'medium':
        return base_n, int(base_steps * 0.5), 0.01
    elif level == 'high':
        return int(base_n * 1.0), base_steps, 0.0
    elif level == 'saturated':
        scale = 1.5
        if profile == 'memory': scale = 1.0 
        return int(base_n * scale), int(base_steps * 1.5), 0.0
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
        
        self.buffer_count = 16 
        self.buffers = ParameterTuple([
            Parameter(Tensor(np.ones((N, N)), self.dtype), name=f"buf_{i}")
            for i in range(self.buffer_count)
        ])
        self.val = Tensor(0.0001, self.dtype)

    def construct(self, _=None):
        last_op = self.val 
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
# 4. Random Load (真实波动模拟 - 增强版)
# =========================
class RandomCell(LoadBase):
    def __init__(self, N, steps):
        super().__init__(N, steps, ms.float16)
        in_channel = N
        # 扩展系数，MobileNetV2通常是6，这里用4保证不过载
        expand_ratio = 4 
        hidden_dim = in_channel * expand_ratio
        
        # [修改点2] 完整的 Inverted Residual Block 结构
        # 1. Expand (1x1 Conv): 升维，高计算密度 -> Cube
        self.conv_pw = nn.Conv2d(in_channel, hidden_dim, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # 2. Depthwise (3x3 Conv): 特征提取，高访存 -> Vector/Memory
        self.conv_dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, 
                                 group=hidden_dim, pad_mode='same', has_bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # 3. Project (1x1 Conv): 降维 -> Cube
        self.conv_proj = nn.Conv2d(hidden_dim, in_channel, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(in_channel)
        
        self.relu = nn.ReLU()
        # Residual add 需要 Vector core
        self.add = ops.Add() 

    def construct(self, x):
        out = x
        for _ in range(self.steps):
            identity = out
            
            # Phase 1: Expand
            out = self.conv_pw(out)
            out = self.bn1(out)
            out = self.relu(out)
            
            # Phase 2: Depthwise
            out = self.conv_dw(out)
            out = self.bn2(out)
            out = self.relu(out)
            
            # Phase 3: Project
            out = self.conv_proj(out)
            out = self.bn3(out)
            
            # Phase 4: Residual Connection
            out = self.add(out, identity)
            
        return self.finish_op(out)

# =========================
# 数据生成器
# =========================
def create_random_input(profile, N):
    if profile != 'random':
        return None
    # 模拟输入 Feature Map (Batch=32, C=N, H=14, W=14)
    return Tensor(np.random.randn(32, N, 14, 14).astype(np.float16))

# =========================
# 主循环
# =========================
def run_loop(net, sleep_interval, profile, N):
    print("Warmup (Compiling graph)...")
    warmup_input = create_random_input(profile, N)
    try:
        if warmup_input is not None:
            net(warmup_input)
        else:
            net(None)
    except Exception as e:
        print(f"Error during warmup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Load started. Base Sleep: {sleep_interval}s")
    
    while True:
        # 1. 准备数据
        current_input = create_random_input(profile, N)
        
        # 2. 执行计算
        if current_input is not None:
            scalar_tensor = net(current_input)
        else:
            scalar_tensor = net(None)
        
        # 3. 同步
        _ = scalar_tensor.asnumpy()
        
        # 4. 休眠控制 (核心修改点)
        if profile == 'random':
            # [修改点3] 真正的随机波动逻辑
            # 在 0 到 sleep_interval * 2 之间随机，模拟负载的不确定性
            # 有时候是 0 (连续高负载)，有时候是 2倍 (长等待)
            wait_time = random.uniform(0, sleep_interval * 2)
            time.sleep(wait_time)
        elif sleep_interval > 0:
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
    print(f"Config: Profile={args.profile}, Level={args.level} -> N={N}, Steps={steps}, Sleep={sleep_interval}s")

    if args.profile == 'compute':
        net = ComputeCell(N, steps)
    elif args.profile == 'memory':
        net = MemoryCell(N, steps)
    elif args.profile == 'data':
        net = DataCell(N, steps)
    elif args.profile == 'random':
        net = RandomCell(N, steps)
    else:
        net = ComputeCell(N, steps)
    
    net.set_train(False)
    run_loop(net, sleep_interval, args.profile, N)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    main()