import time
import math
import random
import threading
from abc import ABC, abstractmethod
from load_config import LoadLevel, ProfileType

class LoadGenerator(ABC):
    def __init__(self):
        self.cur_level = LoadLevel.Idle
        self.running_event = threading.Event()
        self.workers = []

    @abstractmethod
    def start(self, profile: ProfileType, level: LoadLevel):
        pass

    def stop(self):
        self.running_event.clear()
        for w in self.workers:
            # 如果是线程/进程，进行 join
            if hasattr(w, 'join'):
                w.join(timeout=1.0)
            # 如果是进程，可能需要 terminate
            if hasattr(w, 'terminate') and w.is_alive():
                w.terminate()
        self.workers.clear()

# --- 辅助函数：获取模拟波形负载强度 (0.0 - 1.0) ---
def get_wave_intensity(offset_seed: float = 0.0) -> float:
    t = time.time()
    # 1. 基础趋势波 (周期约 60秒)
    slow_wave = 0.5 * math.sin(t / 20.0 + offset_seed)
    # 2. 活跃波动波 (周期约 5秒)
    fast_wave = 0.2 * math.sin(t / 2.5 + offset_seed * 2.0)
    # 3. 随机抖动
    noise = random.uniform(-0.05, 0.05)

    # 结果 [0.1, 0.9]
    val = 0.5 + slow_wave + fast_wave + noise
    return max(0.01, min(val, 1.0))

# --- 辅助函数：根据 Level 获取目标占空比/强度 ---
def get_target_intensity(level: LoadLevel) -> float:
    mapping = {
        LoadLevel.Idle: 0.01,
        LoadLevel.Low: 0.2,
        LoadLevel.Medium: 0.5,
        LoadLevel.High: 0.85,
        LoadLevel.Saturated: 1.0
    }
    return mapping.get(level, 0.1)