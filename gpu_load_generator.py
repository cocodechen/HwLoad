import time
import threading
import torch
from load_config import LoadLevel, ProfileType
from load_generator import LoadGenerator, get_wave_intensity, get_target_intensity

class GPULoadGenerator(LoadGenerator):
    def __init__(self):
        super().__init__()
        self.has_gpu = torch.cuda.is_available()
        if not self.has_gpu:
            print("[GPULoad] Warning: No CUDA device found.")

    def start(self, profile: ProfileType, level: LoadLevel):
        if not self.has_gpu:
            return
            
        self.cur_level = level
        self.running_event.set()

        if profile == ProfileType.Random:
            # Random 模式使用波形函数
            t = threading.Thread(target=self._run_gpu_load, args=(lambda: get_wave_intensity(60),))
        else:
            # 其他模式使用固定 Level
            target = get_target_intensity(level)
            t = threading.Thread(target=self._run_gpu_load, args=(lambda: target,))
        
        t.start()
        self.workers.append(t)

    def _run_gpu_load(self, intensity_func):
        """
        集成 gpu_load_worker.py 的核心逻辑
        """
        device = torch.device("cuda")
        print(f"[GPULoad] Worker started on: {torch.cuda.get_device_name(0)}")

        while self.running_event.is_set():
            target_load = intensity_func()
            
            # 负载极低时，清理显存并休眠
            if target_load < 0.05:
                torch.cuda.empty_cache()
                time.sleep(0.5)
                continue

            # 动态调整矩阵大小
            # 1000 ~ 8000 维方阵
            size = int(1000 + 7000 * target_load)
            
            try:
                # 制造计算压力 (Matmul) 和 显存压力 (Allocation)
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                
                # 循环次数
                loops = int(10 * target_load)
                
                for _ in range(max(1, loops)):
                    if not self.running_event.is_set(): break
                    c = torch.matmul(a, b)
                    # 必须同步才能产生真实的持续占用，否则只会把队列填满立刻返回 python
                    torch.cuda.synchronize() 
                
                # 释放临时变量
                del a, b, c
                
            except Exception as e:
                # 可能是 OOM，休息一下
                torch.cuda.empty_cache()
                time.sleep(1)

            # 休息时间控制占空比
            if target_load < 0.95:
                time.sleep((1.0 - target_load) * 0.2)