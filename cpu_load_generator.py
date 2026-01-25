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
from load_generator import LoadGenerator, get_wave_intensity

class CPULoadGenerator(LoadGenerator):
    def start(self, profile: ProfileType, level: LoadLevel):
        self.cur_level = level
        self.running_event.set()

        # 无论什么 Profile，只要是 Random 模式，就完全按照 load.py 的全套逻辑启动
        if profile == ProfileType.Random:
            self._start_load_py_logic()
        else:
            # 兼容其他固定模式（如果使用了 cpu:compute:high 这种）
            # 这里简单处理，复用同样的 worker 但给固定值
            self._start_fixed_logic(level)

    def _start_load_py_logic(self):
        """完全复刻 load.py 的线程启动逻辑"""
        total_cores = multiprocessing.cpu_count()
        
        # 1. CPU Threads (与 load.py 一模一样)
        for i in range(total_cores):
            offset = random.random() * 10
            # 使用 lambda 捕获 offset
            t = threading.Thread(
                target=self.cpu_worker, 
                args=(lambda o=offset: get_wave_intensity(o), i)
            )
            t.start()
            self.workers.append(t)

        # 2. Memory
        t_mem = threading.Thread(
            target=self.memory_worker, 
            args=(lambda: get_wave_intensity(20),)
        )
        t_mem.start()
        self.workers.append(t_mem)

        # 3. Disk
        t_disk = threading.Thread(
            target=self.disk_worker, 
            args=(lambda: get_wave_intensity(40),)
        )
        t_disk.start()
        self.workers.append(t_disk)

        # 4. Network
        t_net = threading.Thread(
            target=self.network_worker, 
            args=(lambda: get_wave_intensity(80),)
        )
        t_net.start()
        self.workers.append(t_net)

    def _start_fixed_logic(self, level):
        # 辅助兼容函数，如果用户没用 random 模式
        mapping = {LoadLevel.Idle: 0.0, LoadLevel.Low: 0.2, LoadLevel.Medium: 0.5, LoadLevel.High: 0.8, LoadLevel.Saturated: 1.0}
        val = mapping.get(level, 0.5)
        # 启动一个 CPU 线程跑固定负载
        t = threading.Thread(target=self.cpu_worker, args=(lambda: val, 0))
        t.start()
        self.workers.append(t)

    def cpu_worker(self, intensity_func, core_id):
        while self.running_event.is_set():
            target_load = intensity_func()
            if target_load < 0.1:
                time.sleep(0.5) 
                continue
                
            matrix_size = int(1000 * target_load)
            try:
                a = np.random.rand(matrix_size, matrix_size)
                b = np.random.rand(matrix_size, matrix_size)
                np.dot(a, b)
            except Exception:
                pass
            
            if self.running_event.is_set():
                sleep_time = (1.0 - target_load) * 0.1
                time.sleep(max(0.01, sleep_time))

    def memory_worker(self, intensity_func):
        allocated_data = []
        max_memory_percent = 85.0
        while self.running_event.is_set():
            try:
                current_mem_percent = psutil.virtual_memory().percent
                target_load = intensity_func()
                
                if current_mem_percent > max_memory_percent:
                    allocated_data = []
                    time.sleep(1)
                    continue
                    
                target_gb = 100 * target_load 
                current_held_gb = len(allocated_data) * 0.1 
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

    def disk_worker(self, intensity_func):
        disk_file_path = os.path.join(tempfile.gettempdir(), "stress_test_file_py.dat")
        while self.running_event.is_set():
            target_load = intensity_func()
            if target_load < 0.1:
                time.sleep(0.5)
                continue
            
            file_size_mb = int(10 + 490 * target_load)
            block_size = 1024 * 1024
            
            try:
                with open(disk_file_path, "wb") as f:
                    data = os.urandom(block_size)
                    for _ in range(file_size_mb):
                        if not self.running_event.is_set(): break
                        f.write(data)
                
                if not self.running_event.is_set(): break

                if os.path.exists(disk_file_path):
                    with open(disk_file_path, "rb") as f:
                        while f.read(block_size):
                            if not self.running_event.is_set(): break
                
                if os.path.exists(disk_file_path):
                    os.remove(disk_file_path)
            except Exception:
                pass
            
            time.sleep((1.0 - target_load) * 1.0)

    def network_worker(self, intensity_func):
        url = "https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/22.04/ubuntu-22.04.3-desktop-amd64.iso"
        while self.running_event.is_set():
            target_load = intensity_func()
            if target_load < 0.2:
                time.sleep(1)
                continue
            try:
                req = urllib.request.urlopen(url, timeout=3)
                chunk_size = 1024 * 1024 
                read_count = 0
                max_reads = int(50 * target_load)
                while read_count < max_reads and self.running_event.is_set():
                    data = req.read(chunk_size)
                    if not data: break
                    read_count += 1
                req.close()
            except Exception:
                time.sleep(1)
            time.sleep(1)