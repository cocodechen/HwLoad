import os
import time
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

# =========================================================================
#  全参数化负载配置 (SIM_CONFIG)
#  修改这里的参数即可控制所有硬件指标
# =========================================================================
SIM_CONFIG = {
    # ---------------------------------------------------------------------
    # [1. 核心计算维度 - 决定 AICore 和 Vector 的利用率]
    # ---------------------------------------------------------------------
    # N_CHANNELS: 通道数
    # - 影响: 矩阵计算量(Cube) 和 显存占用。
    # - 建议: 2048 是 910B 的甜点。改大(4096)会导致编译极慢；改小(1024)会导致 AICore 吃不饱。
    "N_CHANNELS":      2048,   

    # BATCH_SIZE: 批处理大小
    # - 影响: AICore 利用率 (Cube)。
    # - 增大: 矩阵乘法效率变高，AICore 占用上升。
    # - 减小: 如果生成数据太慢卡住，请减小此值。
    # - 建议: 32 (平衡点)。
    "BATCH_SIZE":      32,     

    # IMG_SIZE: 图片分辨率 (H, W)
    # - 影响: 显存带宽 (MemBW) 和 向量单元 (Vector)。
    # - 增大: 显著增加显存读写压力，显著增加 Vector 负载，显著增加 CPU 生成数据时间。
    # - 建议: 32 (为了秒启动)。如果想测高显存带宽，可改为 64 或 128，但启动会变慢。
    "IMG_SIZE":        32,     

    # INTERNAL_REPEAT: [杀手锏] 单次 Step 内部的重复计算次数
    # - 影响: 纯粹的 NPU 运算时间。
    # - 原理: 数据进去后，在 NPU 内部空转计算多少次才出来。
    # - 增大: AICore/Vector 持续高负载，CPU/IO 压力几乎为 0。
    # - 建议: 50 (保证数据量小也能把算力拉满)。
    "INTERNAL_REPEAT": 30,     

    # ---------------------------------------------------------------------
    # [2. 数据生成配置 - 解决“启动卡死”的关键]
    # ---------------------------------------------------------------------
    # DATA_POOL_COUNT: 内存中准备几组数据 (原代码中的 5)
    # - 影响: CPU 内存占用、启动等待时间。
    # - 说明: NPU 压测不需要多组不同数据，1组反复跑和10组反复跑对 NPU 压力是一样的。
    # - 建议: 1 (秒启动)。如果设为 5 或 10，在大分辨率下会导致 CPU 内存溢出卡死。
    "DATA_POOL_COUNT": 5,

    # ---------------------------------------------------------------------
    # [3. 训练节奏控制]
    # ---------------------------------------------------------------------
    "STEPS_PER_EPOCH": 50,    # 也可以改大，让高负载持续更久
    "EVAL_STEPS":      5,      # 验证阶段步数
    "SAVE_INTERVAL":   5,   # 设大点，避免频繁 IO 打断计算负载
    "CKPT_SIZE_MB":    2048,   # 模拟 IO 写盘大小
}

# =========================
# 1. 混合负载模块 (保持架构不变)
# =========================
class MixedLoadBlock(nn.Cell):
    def __init__(self, in_channel, repeat_times):
        super().__init__()
        self.repeat_times = repeat_times
        
        hidden_dim = in_channel // 2 
        
        # Cube 负载组件
        self.conv_cube_1 = nn.Conv2d(in_channel, hidden_dim, kernel_size=1, has_bias=False)
        self.conv_cube_2 = nn.Conv2d(hidden_dim, in_channel, kernel_size=1, has_bias=False)
        
        # Vector/Mem 负载组件
        self.conv_vec = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, 
                                  group=hidden_dim, pad_mode='same', has_bias=False)
        
        # Vector 负载组件
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn3 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()
        self.add = ops.Add()
        
        self.avg_pool = ops.ReduceMean(keep_dims=False)
        self.classifier = nn.Dense(in_channel, 10) 

        self.to_float(ms.float16)
        self.classifier.to_float(ms.float32)

    def construct(self, x):
        current = x
        
        # 内部循环：通过参数 INTERNAL_REPEAT 控制
        for _ in range(self.repeat_times):
            identity = current
            
            # Cube
            out = self.conv_cube_1(current)
            out = self.bn1(out)
            out = self.relu(out)
            
            # Vector & Memory
            out = self.conv_vec(out)
            out = self.bn2(out)
            out = self.relu(out)
            
            # Cube
            out = self.conv_cube_2(out)
            out = self.bn3(out)
            
            # Vector
            out = self.add(out, identity)
            
            current = out

        pool = self.avg_pool(current, (2, 3))
        logits = self.classifier(pool)
        return logits

# =========================
# 2. 辅助功能 (已参数化)
# =========================
def create_ram_dataset():
    """内存数据池"""
    # 提取所有配置参数
    c = SIM_CONFIG["N_CHANNELS"]
    b = SIM_CONFIG["BATCH_SIZE"]
    h = w = SIM_CONFIG["IMG_SIZE"]
    count = SIM_CONFIG["DATA_POOL_COUNT"] # 提取循环次数
    
    print(f"[Simulator] Allocating RAM Dataset...")
    print(f"            Config: {count} batches of ({b}x{c}x{h}x{h})")
    
    data_pool = []
    label_pool = []
    
    t_start = time.time()
    
    # 使用提取出来的参数控制循环
    for i in range(count):
        # 优化提示：如果这里卡住，说明 BATCH_SIZE * IMG_SIZE 太大，或者 count 太多
        # 保持原本的 np.random.randn (虽然慢，但只要 count=1 且 shape 不大就没问题)
        img = Tensor(np.random.randn(b, c, h, w).astype(np.float16))
        
        lbl = np.zeros((b, 10), dtype=np.float32)
        for k in range(b):
            lbl[k, np.random.randint(0, 10)] = 1.0
            
        label_pool.append(lbl)
        data_pool.append(img)
        
        if (i+1) % 1 == 0:
            print(f"            -> Generated batch {i+1}/{count}")

    print(f"[Simulator] Dataset ready in {time.time()-t_start:.2f}s")
    return data_pool, label_pool
        
def simulate_checkpoint_io():
    size_mb = SIM_CONFIG["CKPT_SIZE_MB"]
    if size_mb <= 0: return
    print(f" [IO] Saving Checkpoint ({size_mb} MB)...", end="", flush=True)
    filename = "temp_stress_test.ckpt"
    chunk = os.urandom(10 * 1024 * 1024) 
    num_chunks = size_mb // 10
    with open(filename, "wb") as f:
        for _ in range(num_chunks):
            f.write(chunk)
            os.fsync(f.fileno())
    if os.path.exists(filename): os.remove(filename)
    print(" Done.")

# =========================
# 3. 主程序
# =========================
def run_simulation_loop():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
    
    print(f"\n>>> Starting Parametrized Load Simulator <<<")
    print(f">>> Config: {SIM_CONFIG}\n")

    # 初始化模型
    net = MixedLoadBlock(
        in_channel=SIM_CONFIG["N_CHANNELS"], 
        repeat_times=SIM_CONFIG["INTERNAL_REPEAT"]
    )
    net.set_train(True)
    
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
    train_net = nn.TrainOneStepCell(nn.WithLossCell(net, loss_fn), opt)
    
    # 准备数据
    data_pool, label_pool = create_ram_dataset()
    pool_len = len(data_pool)
    
    # Warmup
    print("[Simulator] Compiling graph (Wait ~30-60s)...")
    t_compile = time.time()
    _ = train_net(data_pool[0], Tensor(label_pool[0]))
    print(f"[Simulator] Compiled in {time.time()-t_compile:.2f}s")
    
    epoch = 0
    while True:
        epoch += 1
        print(f"=== Epoch {epoch} ===")
        
        # Stage 1: Training
        t_start = time.time()
        steps = SIM_CONFIG["STEPS_PER_EPOCH"]
        
        for i in range(steps):
            idx = i % pool_len
            loss = train_net(data_pool[idx], Tensor(label_pool[idx]))
            
            if i % 10 == 0:
                print(f"\r  [Train] Step {i}/{steps} | Loss: {float(loss.asnumpy()):.4f}", end="")
        
        duration = time.time() - t_start
        # 计算等效 FPS (考虑内部循环)
        total_ops_factor = steps * SIM_CONFIG["INTERNAL_REPEAT"]
        print(f"\r  [Train] Finished. Time: {duration:.2f}s")
        
        # Stage 2: Checkpoint
        if epoch % SIM_CONFIG["SAVE_INTERVAL"] == 0:
            simulate_checkpoint_io()
