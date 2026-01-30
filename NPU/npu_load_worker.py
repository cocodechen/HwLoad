import argparse
import time
import signal
import sys
import random
import gc
import numpy as np

import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor, context, Parameter, ParameterTuple

# 尝试导入真实训练模拟器
try:
    from npu_train import run_simulation_loop
except ImportError:
    run_simulation_loop = None

# ============================================================================
# 系统信号处理
# ============================================================================
def signal_handler(sig, frame):
    print("\n[System] Received stop signal, exiting gracefully...")
    sys.exit(0)

# ============================================================================
# 配置管理 (Refactored)
# ============================================================================
def get_config(profile: str, level: str):
    """
    根据负载类型(profile)和压力等级(level)返回具体参数。
    使用字典查表方式，方便手动微调特定档位的参数。
    
    Returns:
        tuple: (Matrix_Size_N, Steps_Per_Loop, Sleep_Interval)
    """
    
    # 基础参数配置表
    # 格式: 'Level': (N, Steps, Sleep)
    
    # 1. Compute: 算力密集型 (侧重矩阵乘法计算量)
    #    Base: 4096, Steps: 200
    compute_configs = {
        'saturated': (8192, 100, 0.0),   # 极致压力：超大矩阵，无休眠
        'high':      (4096, 200, 0.0),   # 高压力：标准大矩阵，长步数
        'medium':    (4096, 100, 0.01),  # 中等：步数减半，微量休眠
        'low':       (4096, 40,  0.05),  # 低压：短步数，明显休眠
        'idle':      (64,   1,   1.0),   # 空转：极小计算，长休眠
    }

    # 2. Memory: 带宽密集型 (侧重显存块读写)
    #    注意：N在此处决定了显存块的大小 (8192=256MB, 4096=64MB, 2048=16MB)
    memory_configs = {
        'saturated': (8192, 20, 0.0),    # 极致带宽：256MB大块拷贝
        'high':      (4096, 40, 0.0),    # 高带宽：64MB块
        'medium':    (2048, 50, 0.05),   # 中等：16MB块，带间歇
        'low':       (2048, 20, 0.1),    # 低压
        'idle':      (64,   1,  1.0),
    }

    # 3. Data: 数据搬运型 (侧重类型转换 Cast 和数据流动)
    #    Base: 2048, Steps: 50
    data_configs = {
        'saturated': (3072, 75, 0.0),    # 1.5x Base
        'high':      (2048, 50, 0.0),    # 1.0x Base
        'medium':    (2048, 25, 0.01),   # 0.5x Base
        'low':       (2048, 10, 0.05),   # 0.2x Base
        'idle':      (64,   1,  1.0),
    }

    # 4. Train: 模拟真实训练 (参数固定，波动由内部控制)
    train_config = (1024, 15, 0.0)

    # --- 查表逻辑 ---
    
    if profile == 'train':
        return train_config
    
    # 默认回退配置
    default_config = (1024, 10, 0.1)
    
    if profile == 'compute':
        return compute_configs.get(level, default_config)
    elif profile == 'memory':
        return memory_configs.get(level, default_config)
    elif profile == 'data':
        return data_configs.get(level, default_config)
    else:
        # 如果传入未知的 profile，默认使用 compute 的配置
        return compute_configs.get(level, default_config)

# ============================================================================
# 负载单元定义 (MindSpore Cells)
# ============================================================================
class LoadBase(nn.Cell):
    """负载基类，处理通用的结果规约"""
    def __init__(self, N, steps, dtype):
        super().__init__()
        self.steps = steps
        self.dtype = dtype
        self.reduce = ops.ReduceSum()

    def finish_op(self, x):
        return self.reduce(x)

class ComputeCell(LoadBase):
    """计算密集型 (FP16 MatMul) - 压测 Cube Core"""
    def __init__(self, N, steps):
        super().__init__(N, steps, ms.float16)
        self.matmul = ops.MatMul()
        # 初始化随机矩阵
        self.x = Parameter(Tensor(np.random.normal(0, 0.01, (N, N)), self.dtype), name="x")
        self.y = Parameter(Tensor(np.random.normal(0, 0.01, (N, N)), self.dtype), name="y")

    def construct(self, _=None):
        out = self.x
        for _ in range(self.steps):
            out = self.matmul(out, self.y)
        return self.finish_op(out)

class MemoryCell(LoadBase):
    """显存带宽密集型 (FP32 Copy/Add) - 压测 HBM"""
    def __init__(self, N, steps):
        super().__init__(N, steps, ms.float32)
        self.add = ops.Add()
        self.assign = ops.Assign()
        self.depend = ops.Depend()
        
        # 根据矩阵大小动态调整 Buffer 数量，防止 OOM 同时保证压力
        if N >= 8192:
            self.buffer_count = 8   # ~2GB (8 * 256MB)
        elif N >= 4096:
            self.buffer_count = 16  # ~1GB (16 * 64MB)
        else:
            self.buffer_count = 32  # ~512MB
            
        mem_usage = self.buffer_count * N * N * 4 / (1024**2)
        print(f"[Memory Load] Allocating {self.buffer_count} blocks of {N}x{N} FP32 (Total: ~{mem_usage:.2f} MB)")
        
        self.buffers = ParameterTuple([
            Parameter(Tensor(np.ones((N, N)), self.dtype), name=f"buf_{i}")
            for i in range(self.buffer_count)
        ])
        self.val = Tensor(0.0001, self.dtype)

    def construct(self, _=None):
        last_op = self.val 
        # 链式依赖，强制串行读写
        for i in range(self.buffer_count):
            buf = self.buffers[i]
            current_buf = self.depend(buf, last_op)
            res = self.add(current_buf, self.val)
            last_op = self.assign(buf, res)
        return self.finish_op(self.buffers[0])

class DataCell(LoadBase):
    """数据搬运模拟 (FP32 <-> FP16 Cast) - 混合压力"""
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

# ============================================================================
# 执行逻辑
# ============================================================================

def create_net(profile, N, steps):
    """工厂方法：创建负载网络"""
    if profile == 'compute':
        return ComputeCell(N, steps)
    elif profile == 'memory':
        return MemoryCell(N, steps)
    elif profile == 'data':
        return DataCell(N, steps)
    else:
        return ComputeCell(N, steps)

def run_fixed_loop(net, sleep_interval, profile):
    """固定配置运行循环"""
    print(f"Warmup (Compiling graph)...")
    try:
        net(None) # 触发图编译
    except Exception as e:
        print(f"Error during warmup: {e}")
        sys.exit(1)
    
    print(f"Load started. Profile: {profile} (Running FULL SPEED)")
    
    while True:
        net(None)
        if sleep_interval > 0:
            time.sleep(sleep_interval)

def run_random_schedule():
    """
    随机混合调度模式：
    周期性地在不同负载类型和压力等级之间切换，模拟无规律的集群环境。
    """
    available_profiles = ['compute', 'memory', 'data']
    available_levels = ['low', 'medium', 'high', 'saturated']
    
    print(">>> Starting RANDOM schedule mode (Mixed Workload) <<<")
    
    while True:
        # 1. 随机决策
        curr_profile = random.choice(available_profiles)
        curr_level = random.choice(available_levels)
        duration = random.randint(10, 30) # 保持 10~30秒
        
        print(f"\n[Scheduler] Switch -> {curr_profile.upper()} - {curr_level.upper()} (Duration: {duration}s)")
        
        # 2. 获取配置
        N, steps, sleep_interval = get_config(curr_profile, curr_level)
        
        # 3. 构建网络 (重建 Net 会触发 MindSpore 重新编译图)
        net = create_net(curr_profile, N, steps)
        net.set_train(False)
        
        # 4. 执行阶段
        seg_start = time.time()
        iter_count = 0
        
        try:
            # 首次运行触发编译
            net(None) 
            
            while time.time() - seg_start < duration:
                net(None)
                iter_count += 1
                if sleep_interval > 0:
                    time.sleep(sleep_interval)
                    
        except Exception as e:
            print(f"  [Error] Execution failed: {e}")
        
        print(f"  -> Done. Executed {iter_count} iters.")
        
        # 5. 清理资源
        del net
        gc.collect() # 显式 GC，确保显存及时释放

def main():
    parser = argparse.ArgumentParser(description="NPU Load Generator")
    parser.add_argument('--profile', type=str, default='train', 
                        help='train (SIMULATION), random (MIXED SCHEDULE), compute, memory, data')
    parser.add_argument('--level', type=str, default='high', 
                        help='idle/low/medium/high/saturated')
    args = parser.parse_args()

    # 初始化 Context
    try:
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    except Exception as e:
        print(f"Init Context Failed: {e}")
        sys.exit(1)

    # 注册信号
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # --- 模式分发 ---
    
    # 1. Train 模式: 调用外部 npu_train 模块
    if args.profile == 'real':
        if run_simulation_loop is None:
            print("Error: 'npu_train.py' not found.")
            sys.exit(1)
        run_simulation_loop()
        
    # 2. Random 模式: 内部混合调度
    elif args.profile == 'random':
        run_random_schedule()
        
    # 3. Fixed 模式: 单一负载持续运行
    else:
        N, steps, sleep_interval = get_config(args.profile, args.level)
        print(f"Config: Profile={args.profile}, Level={args.level} -> N={N}, Steps={steps}")

        net = create_net(args.profile, N, steps)
        net.set_train(False)
        run_fixed_loop(net, sleep_interval, args.profile)

if __name__ == "__main__":
    main()