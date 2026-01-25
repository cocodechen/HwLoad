import time
import os
import random
import threading
import multiprocessing
import psutil
import tempfile
import numpy as np
import urllib.request
from load_config import LoadLevel, ProfileType
from load_generator import LoadGenerator, get_wave_intensity, get_target_intensity

# --- 核心计算函数 (放在类外以方便 multiprocessing 调用) ---
def _cpu_compute_task(running_event, intensity_func_wrapper):
    """纯计算密集型任务：矩阵乘法"""
    while running_event.is_set():
        # intensity_func_wrapper 可以是一个返回固定值的函数，也可以是 wave 函数
        target_load = intensity_func_wrapper()
        
        if target_load < 0.05:
            time.sleep(0.5)
            continue
            
        # 动态调整矩阵大小以控制计算量
        # target 1.0 -> 大约耗时几百毫秒
        matrix_size = int(200 + 400 * target_load) 
        
        start_t = time.time()
        try:
            # Numpy 在做矩阵运算时通常会释放 GIL
            a = np.random.rand(matrix_size, matrix_size)
            b = np.random.rand(matrix_size, matrix_size)
            np.dot(a, b)
        except Exception:
            pass
        elapsed = time.time() - start_t

        # 简单的占空比控制：工作 elapsed 秒，需要休息 sleep 秒
        # Load = Work / (Work + Sleep) => Sleep = Work * (1/Load - 1)
        if target_load < 0.99:
            sleep_time = elapsed * (1.0/target_load - 1.0)
            time.sleep(max(0.0, sleep_time))

class CPULoadGenerator(LoadGenerator):
    def start(self, profile: ProfileType, level: LoadLevel):
        self.cur_level = level
        self.running_event.set()

        if profile == ProfileType.Compute:
            self._start_compute()
        elif profile == ProfileType.Memory:
            self._start_memory()
        elif profile == ProfileType.IO:
            self._start_io()
        elif profile == ProfileType.Random:
            self._start_random() # 集成 cpu_load_worker 逻辑
        else:
            # 默认混合 Data 模式
            self._start_compute() 
            
    def _start_compute(self):
        # 使用多进程占满所有核
        num_cores = multiprocessing.cpu_count()
        intensity = get_target_intensity(self.cur_level)
        
        for _ in range(num_cores):
            p = multiprocessing.Process(
                target=_cpu_compute_task,
                args=(self.running_event, lambda: intensity)
            )
            p.start()
            self.workers.append(p)

    def _start_memory(self):
        # 内存占用线程
        t = threading.Thread(target=self._run_memory_burn, args=(lambda: get_target_intensity(self.cur_level),))
        t.start()
        self.workers.append(t)

    def _start_io(self):
        # IO 压力线程
        t = threading.Thread(target=self._run_disk_burn, args=(lambda: get_target_intensity(self.cur_level),))
        t.start()
        self.workers.append(t)

    def _start_random(self):
        """
        [NEW] Random Profile: 对应原来的 cpu_load_worker.py
        混合 CPU, Memory, Disk, Network，且负载呈波形变化
        """
        num_cores = multiprocessing.cpu_count()
        
        # 1. CPU 线程 (原代码使用 threading，对于 Random 模式我们保持原样，
        # 但为了效果更好，推荐这里如果是 Random 也可以用 Process)
        # 既然要求移植 cpu_load_worker，它原来是用 threading + numpy
        for i in range(num_cores):
            offset = random.random() * 10
            # 使用波形函数
            t = threading.Thread(
                target=self._run_cpu_numpy_burn, 
                args=(lambda: get_wave_intensity(offset),)
            )
            t.start()
            self.workers.append(t)

        # 2. Memory
        t_mem = threading.Thread(target=self._run_memory_burn, args=(lambda: get_wave_intensity(20),))
        t_mem.start()
        self.workers.append(t_mem)

        # 3. Disk
        t_disk = threading.Thread(target=self._run_disk_burn, args=(lambda: get_wave_intensity(40),))
        t_disk.start()
        self.workers.append(t_disk)

        # 4. Network
        t_net = threading.Thread(target=self._run_network_burn, args=(lambda: get_wave_intensity(80),))
        t_net.start()
        self.workers.append(t_net)

    # --- Worker Implementations ---

    def _run_cpu_numpy_burn(self, intensity_func):
        # 对应 cpu_load_worker 中的 cpu_worker
        while self.running_event.is_set():
            target_load = intensity_func()
            if target_load < 0.1:
                time.sleep(0.5)
                continue
            
            matrix_size = int(300 * target_load) # 稍微减小尺寸适应多线程
            try:
                a = np.random.rand(matrix_size, matrix_size)
                b = np.random.rand(matrix_size, matrix_size)
                np.dot(a, b)
            except Exception:
                pass
            
            sleep_time = (1.0 - target_load) * 0.1
            time.sleep(max(0.01, sleep_time))

    def _run_memory_burn(self, intensity_func):
        # 对应 cpu_load_worker 中的 memory_worker
        allocated_data = []
        max_mem_percent = 85.0
        while self.running_event.is_set():
            try:
                current_mem = psutil.virtual_memory().percent
                target_load = intensity_func()

                # 保护机制
                if current_mem > max_mem_percent:
                    allocated_data = [] # 释放
                    time.sleep(1)
                    continue
                
                # 目标：假设最大测试占用 10GB * target_load (或者基于系统总内存)
                # 这里简单化：target_load * 100个 chunk
                # 调整：动态申请和释放
                chunk_size = 50 * 1024 * 1024 # 50MB
                target_chunks = int(20 * target_load) # 最多约 1GB 动态波动

                if len(allocated_data) < target_chunks:
                    try:
                        allocated_data.append(bytearray(os.urandom(chunk_size)))
                    except MemoryError:
                        time.sleep(1)
                elif len(allocated_data) > target_chunks:
                    if allocated_data:
                        allocated_data.pop()
                
                time.sleep(0.2)
            except Exception:
                time.sleep(0.5)

    def _run_disk_burn(self, intensity_func):
        # 对应 cpu_load_worker 中的 disk_worker
        disk_path = os.path.join(tempfile.gettempdir(), "stress_test_py_io.dat")
        while self.running_event.is_set():
            target_load = intensity_func()
            if target_load < 0.1:
                time.sleep(0.5)
                continue
            
            # 写
            file_size_mb = int(10 + 100 * target_load)
            block_size = 1024 * 1024
            try:
                with open(disk_path, "wb") as f:
                    data = os.urandom(block_size)
                    for _ in range(file_size_mb):
                        if not self.running_event.is_set(): break
                        f.write(data)
                
                # 读
                if os.path.exists(disk_path) and self.running_event.is_set():
                    with open(disk_path, "rb") as f:
                        while f.read(block_size):
                            if not self.running_event.is_set(): break
                
                # 清理
                if os.path.exists(disk_path):
                    os.remove(disk_path)
            except Exception:
                pass
            
            # 休息
            time.sleep((1.0 - target_load) * 0.5)

    def _run_network_burn(self, intensity_func):
        # 对应 cpu_load_worker 中的 network_worker
        # 注意：不要在生产环境攻击公共镜像源，这里使用 dummy url 或较小的文件
        url = "http://example.com" 
        while self.running_event.is_set():
            target_load = intensity_func()
            if target_load < 0.3:
                time.sleep(1)
                continue
            try:
                req = urllib.request.urlopen(url, timeout=2)
                req.read()
                req.close()
            except Exception:
                pass
            # 频率控制
            time.sleep(max(0.1, (1.0 - target_load)))