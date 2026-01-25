import time
import random
import math
import multiprocessing
import threading
import os
import psutil
import tempfile
import numpy as np
import sys

# 尝试导入 PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class LoadController:
    def __init__(self):
        self.running = True
        self.max_memory_percent = 85.0
        self.disk_file_path = os.path.join(tempfile.gettempdir(), "stress_test_file.dat")
        self.total_cores = multiprocessing.cpu_count()
        
    def stop(self):
        self.running = False

    # --- 1. CPU ---
    def cpu_worker(self, intensity_func, core_id):
        while self.running:
            target_load = intensity_func()
            if target_load < 0.1:
                time.sleep(0.5) # 缩短休眠以便更快响应退出信号
                continue
                
            matrix_size = int(500 * target_load)
            try:
                a = np.random.rand(matrix_size, matrix_size)
                b = np.random.rand(matrix_size, matrix_size)
                np.dot(a, b)
            except Exception:
                pass
            
            # 动态休眠
            if self.running:
                sleep_time = (1.0 - target_load) * 0.1
                time.sleep(max(0.01, sleep_time))

    # --- 2. Memory ---
    def memory_worker(self, intensity_func):
        allocated_data = []
        while self.running:
            try:
                current_mem_percent = psutil.virtual_memory().percent
                target_load = intensity_func()
                
                if current_mem_percent > self.max_memory_percent:
                    allocated_data = []
                    time.sleep(1)
                    continue
                    
                target_gb = 100 * target_load 
                # 估算当前持有的内存
                current_held_gb = len(allocated_data) * 0.1 # 100MB per chunk
                
                chunk_size = 100 * 1024 * 1024 
                
                if current_held_gb < target_gb:
                    try:
                        allocated_data.append(bytearray(os.urandom(chunk_size))) 
                    except MemoryError:
                        time.sleep(1)
                elif current_held_gb > target_gb + 2:
                    if allocated_data:
                        allocated_data.pop()
                
                time.sleep(0.5)
            except Exception:
                time.sleep(0.5)

    # --- 3. Disk ---
    def disk_worker(self, intensity_func):
        while self.running:
            target_load = intensity_func()
            if target_load < 0.1:
                time.sleep(0.5)
                continue
            
            file_size_mb = int(10 + 490 * target_load)
            block_size = 1024 * 1024
            
            try:
                # 写
                with open(self.disk_file_path, "wb") as f:
                    data = os.urandom(block_size)
                    for _ in range(file_size_mb):
                        if not self.running: break
                        f.write(data)
                
                if not self.running: break

                # 读
                if os.path.exists(self.disk_file_path):
                    with open(self.disk_file_path, "rb") as f:
                        while f.read(block_size):
                            if not self.running: break
                
                # 删
                if os.path.exists(self.disk_file_path):
                    os.remove(self.disk_file_path)

            except Exception:
                # 忽略IO错误，防止退出时报错
                pass
            
            time.sleep((1.0 - target_load) * 1.0)

    # --- 4. GPU ---
    def gpu_worker(self, intensity_func):
        if not HAS_TORCH or not torch.cuda.is_available():
            return

        device = torch.device("cuda")
        print(f"GPU Load started on: {torch.cuda.get_device_name(0)}")
        
        while self.running:
            target_load = intensity_func()
            if target_load < 0.1:
                torch.cuda.empty_cache()
                time.sleep(0.5)
                continue

            size = int(1000 + 7000 * target_load)
            
            try:
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                # 减少单次循环次数，增加检查 running 标志的频率
                loops = int(5 * target_load) 
                for _ in range(max(1, loops)):
                    if not self.running: break
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
            except Exception:
                pass

            time.sleep((1.0 - target_load) * 0.2)

    # --- 5. Network ---
    def network_worker(self, intensity_func):
        # 简单模拟，如果不通网可以注释掉
        import urllib.request
        url = "https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/22.04/ubuntu-22.04.3-desktop-amd64.iso"
        
        while self.running:
            target_load = intensity_func()
            if target_load < 0.2:
                time.sleep(1)
                continue
            
            try:
                req = urllib.request.urlopen(url, timeout=3)
                chunk_size = 1024 * 1024 
                read_count = 0
                max_reads = int(50 * target_load)
                
                while read_count < max_reads and self.running:
                    data = req.read(chunk_size)
                    if not data: break
                    read_count += 1
                req.close()
            except Exception:
                time.sleep(1)
            time.sleep(1)

    def get_wave_intensity(self, offset):
        t = time.time()
        slow = (math.sin(t / 60.0 + offset) + 1) / 2 
        fast = (math.sin(t / 5.0 + offset * 3) + 1) / 2
        noise = random.random()
        val = 0.5 * slow + 0.3 * fast + 0.2 * noise
        return min(max(val, 0.0), 1.0)

if __name__ == "__main__":
    controller = LoadController()
    threads = []
    
    print(f"Starting Stress Test on {controller.total_cores} Cores, 187GB RAM, Tesla P100...")
    print("Press Ctrl+C to stop.")

    # 启动线程，不再使用 daemon=True，而是手动管理退出
    try:
        # CPU
        for i in range(controller.total_cores):
            offset = random.random() * 10
            t = threading.Thread(target=controller.cpu_worker, args=(lambda: controller.get_wave_intensity(offset), i))
            t.start()
            threads.append(t)

        # Mem
        t_mem = threading.Thread(target=controller.memory_worker, args=(lambda: controller.get_wave_intensity(20),))
        t_mem.start()
        threads.append(t_mem)

        # Disk
        t_disk = threading.Thread(target=controller.disk_worker, args=(lambda: controller.get_wave_intensity(40),))
        t_disk.start()
        threads.append(t_disk)

        # GPU
        t_gpu = threading.Thread(target=controller.gpu_worker, args=(lambda: controller.get_wave_intensity(60),))
        t_gpu.start()
        threads.append(t_gpu)

        # Net
        t_net = threading.Thread(target=controller.network_worker, args=(lambda: controller.get_wave_intensity(80),))
        t_net.start()
        threads.append(t_net)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping load generator... (waiting for threads to finish)")
        controller.stop() # 1. 设置标志位
        
        # 2. 等待所有线程安全结束 (Join)
        for t in threads:
            t.join(timeout=2.0) # 最多等2秒，不等死锁
            
        # 3. 清理临时文件
        if os.path.exists(controller.disk_file_path):
            try:
                os.remove(controller.disk_file_path)
            except:
                pass
        print("Done.")
        sys.exit(0)