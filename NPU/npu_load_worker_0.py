import argparse
import time
import signal
import sys
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor, context, Parameter

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
    """
    策略：
    1. 通过 N (矩阵大小) 控制显存占用量。
    2. 通过 Steps (图内循环) 控制单次 Launch 的 NPU 占用时长。
    3. 通过 Sleep (图间休眠) 控制整体利用率 (Duty Cycle)。
    """
    # === 基础参数 ===
    if profile == 'compute':
        # MatMul 算力极强，需要较大 Steps 减少 launch 开销
        base_n = 4096
        base_steps = 200 
    elif profile == 'memory':
        # FP32 占用显存大，steps 适中
        base_n = 4096 
        base_steps = 50 
    else: # data
        base_n = 2048
        base_steps = 50

    # === Level 调节 (Duty Cycle) ===
    # 通过调节 sleep_interval 实现不同 Level 的指标正相关
    if level == 'idle':
        return 64, 1, 1.0
    elif level == 'low':
        # 跑 50ms, 歇 50ms -> 约 50% 利用率 (或更低，取决于 step)
        return base_n, int(base_steps * 0.2), 0.05
    elif level == 'medium':
        # 跑 100ms, 歇 10ms
        return base_n, int(base_steps * 0.5), 0.01
    elif level == 'high':
        # 连续跑
        return int(base_n * 1.2), base_steps, 0.0
    elif level == 'saturated':
        # 加大显存和计算量
        # 针对 Memory 场景，Scale 需要克制防止 OOM
        scale = 1.5 if profile == 'memory' else 2.0
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
        self.reduce = ops.ReduceSum() # 用于最后坍缩结果

    def finish_op(self, x):
        """
        关键优化：
        将大矩阵坍缩为标量。
        这样 .asnumpy() 只拷贝 4 字节，消除了 PCIe 带宽干扰，
        同时依然起到了同步 Barrier 的作用。
        """
        return self.reduce(x)

# =========================
# Compute Load (计算密集)
# =========================
class ComputeCell(LoadBase):
    def __init__(self, N, steps):
        super().__init__(N, steps, ms.float16)
        self.matmul = ops.MatMul()
        # 初始化非零值，避免被优化
        self.x = Parameter(Tensor(np.random.normal(0, 0.01, (N, N)), self.dtype), name="x")
        self.y = Parameter(Tensor(np.random.normal(0, 0.01, (N, N)), self.dtype), name="y")

    def construct(self):
        out = self.x
        # 纯计算链，不写回 Parameter，最大化 Cube 利用率
        for _ in range(self.steps):
            out = self.matmul(out, self.y)
        
        return self.finish_op(out)
    
# =========================
# Memory Load (访存密集)
# =========================
class MemoryCell(LoadBase):
    def __init__(self, N, steps):
        super().__init__(N, steps, ms.float32) # FP32 带宽压力更大
        self.add = ops.Add()
        self.assign = ops.Assign()
        self.depend = ops.Depend() # <--- 新增 Depend 算子
        
        # 定义 Parameter 以便进行 Assign 写回
        self.data = Parameter(Tensor(np.ones((N, N)), self.dtype), name="data")
        # 用于加法的常量
        self.val = Tensor(0.00001, self.dtype)

    def construct(self):
        # 使用 current 维持循环的数据依赖
        current = self.data
        
        for _ in range(self.steps):
            # 1. 计算 (Read + Compute)
            res = self.add(current, self.val)
            
            # 2. 写回 (Write HBM)
            # 执行 Assign 操作，但这行代码本身在图中可能被忽略，如果后续没有用到 assign_op
            assign_op = self.assign(self.data, res)
            
            # 3. 依赖控制 (Standard Fix)
            # 这句的意思是：返回 res 的值给 current，但前提是 assign_op 必须先执行完
            # 这样既避免了使用 Assign 的返回值（消除了警告），又保证了写操作一定发生
            current = self.depend(res, assign_op)
            
        return self.finish_op(current)

# =========================
# Data/Mixed Load (混合)
# =========================
class DataCell(LoadBase):
    def __init__(self, N, steps):
        super().__init__(N, steps, ms.float32)
        self.matmul = ops.MatMul()
        self.cast = ops.Cast()
        self.src = Parameter(Tensor(np.random.normal(0, 1, (N, N)), ms.float32), name="src")
        self.weight = Parameter(Tensor(np.random.normal(0, 1, (N, N)), ms.float16), name="w")

    def construct(self):
        x = self.src
        w = self.weight
        for _ in range(self.steps):
            # 模拟混合精度训练流程：Cast(Read/Write) + MatMul(Compute)
            x_16 = self.cast(x, ms.float16)
            res = self.matmul(x_16, w)
            x = self.cast(res, ms.float32)
        
        return self.finish_op(x)

# =========================
# 主循环
# =========================
def run_loop(net, sleep_interval):
    print("Warmup (Compiling graph)...")
    try:
        # 第一次运行会触发图编译，耗时较长
        net()
    except Exception as e:
        print(f"Error during warmup: {e}")
        sys.exit(1)
    
    print(f"Load started. Sleep interval: {sleep_interval}s")
    
    while True:
        # 1. 触发执行
        scalar_tensor = net()
        
        # 2. 同步 Barrier
        # 由于 scalar_tensor 只是一个标量，这里的数据传输微乎其微。
        # 但它强制 CPU 等待 NPU 执行完毕。
        _ = scalar_tensor.asnumpy()
        
        # 3. 占空比控制 (Duty Cycle)
        # 只有在 NPU 真正跑完后，我们才开始 sleep，确保了 load/idle 比例的准确性
        if sleep_interval > 0:
            time.sleep(sleep_interval)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=str, default='compute', help='compute/memory/data')
    parser.add_argument('--level', type=str, default='low', help='idle/low/medium/high/saturated')
    args = parser.parse_args()

    # 初始化环境
    try:
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    except Exception as e:
        print(f"Init Context Failed: {e}")
        sys.exit(1)

    # 获取参数
    N, steps, sleep_interval = get_config(args.profile, args.level)
    print(f"Config: Profile={args.profile}, Level={args.level} -> N={N}, Steps={steps}, Sleep={sleep_interval}s")

    # 实例化网络
    if args.profile == 'compute':
        net = ComputeCell(N, steps)
    elif args.profile == 'memory':
        net = MemoryCell(N, steps)
    elif args.profile == 'data':
        net = DataCell(N, steps)
    else:
        net = ComputeCell(N, steps)
    
    net.set_train(False)
    # 运行
    run_loop(net, sleep_interval)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    main()