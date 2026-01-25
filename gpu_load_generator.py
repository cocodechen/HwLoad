import time
import random
import threading
import torch
from load_config import LoadLevel, ProfileType
from load_generator import LoadGenerator, get_wave_intensity

# --- 复用 load.py 的波形函数 ---
def get_wave_intensity(offset):
    import math
    t = time.time()
    slow = (math.sin(t / 60.0 + offset) + 1) / 2 
    fast = (math.sin(t / 5.0 + offset * 3) + 1) / 2
    noise = random.random()
    val = 0.5 * slow + 0.3 * fast + 0.2 * noise
    return min(max(val, 0.0), 1.0)

class GPULoadGenerator(LoadGenerator):
    def __init__(self):
        super().__init__()
        self.has_gpu = False
        try:
            if torch.cuda.is_available():
                # 简单测试防止后续报错
                torch.zeros(1).cuda()
                self.has_gpu = True
            else:
                print("[GPULoad] Warning: No CUDA device detected.")
        except Exception:
            self.has_gpu = False

    def start(self, profile: ProfileType, level: LoadLevel):
        if not self.has_gpu: return
        self.cur_level = level
        self.running_event.set()

        if profile == ProfileType.Random:
            # Random 模式：使用 load.py 的参数 (offset=60)
            t = threading.Thread(
                target=self.gpu_worker, 
                args=(lambda: get_wave_intensity(60),)
            )
        else:
            # 兼容其他固定模式
            val = 0.5
            if level == LoadLevel.High: val = 0.8
            elif level == LoadLevel.Saturated: val = 1.0
            t = threading.Thread(target=self.gpu_worker, args=(lambda: val,))
        
        t.start()
        self.workers.append(t)

    def gpu_worker(self, intensity_func):
        device = torch.device("cuda")
        print(f"[GPULoad] Worker started on: {torch.cuda.get_device_name(0)}")
        
        while self.running_event.is_set():
            target_load = intensity_func()
            if target_load < 0.1:
                torch.cuda.empty_cache()
                time.sleep(0.5)
                continue

            # [关键] 保持 load.py 的 1000 + 7000
            size = int(1000 + 7000 * target_load)
            
            try:
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                # [关键] 保持 load.py 的 loops = 5 * load
                loops = int(5 * target_load) 
                for _ in range(max(1, loops)):
                    if not self.running_event.is_set(): break
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
            except Exception:
                pass

            # [关键] 保持 load.py 的休眠公式
            time.sleep((1.0 - target_load) * 0.2)