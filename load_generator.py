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
def get_wave_intensity(offset):
    import math
    t = time.time()
    slow = (math.sin(t / 60.0 + offset) + 1) / 2 
    fast = (math.sin(t / 5.0 + offset * 3) + 1) / 2
    noise = random.random()
    val = 0.5 * slow + 0.3 * fast + 0.2 * noise
    return min(max(val, 0.0), 1.0)

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