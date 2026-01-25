import sys
import time
import signal
import multiprocessing
from typing import List

from load_config import ProfileType, LoadLevel, LoadTask
from load_generator import LoadGenerator
from cpu_load_generator import CPULoadGenerator
try:
    from gpu_load_generator import GPULoadGenerator
except ImportError:
    GPULoadGenerator = None

stop_requested = False

def signal_handler(signum, frame):
    global stop_requested
    print("\n[Main] Stop signal received.")
    stop_requested = True

def parse_profile(s: str) -> ProfileType:
    s = s.lower()
    mapping = {
        "compute": ProfileType.Compute,
        "memory": ProfileType.Memory,
        "io": ProfileType.IO,
        "data": ProfileType.Data,
        "random": ProfileType.Random
    }
    if s not in mapping:
        raise ValueError(f"Unknown profile: {s}")
    return mapping[s]

def parse_level(s: str) -> LoadLevel:
    s = s.lower()
    mapping = {
        "idle": LoadLevel.Idle,
        "low": LoadLevel.Low,
        "medium": LoadLevel.Medium,
        "high": LoadLevel.High,
        "saturated": LoadLevel.Saturated
    }
    if s not in mapping:
        raise ValueError(f"Unknown level: {s}")
    return mapping[s]

def parse_arg(arg: str) -> LoadTask:
    parts = arg.split(':')
    device = parts[0].lower()
    
    if len(parts) == 2:
        # device:random
        profile = parse_profile(parts[1])
        if profile == ProfileType.Random:
            # Random 模式下 Level 不重要，给个默认值，但在打印时我们会特殊处理
            return LoadTask(device, profile, LoadLevel.Medium)
        else:
            raise ValueError(f"Profile '{parts[1]}' requires a level.")
    elif len(parts) == 3:
        profile = parse_profile(parts[1])
        # 如果用户显式指定 random:high，我们也接受，但实际上内部逻辑是波动的
        level = parse_level(parts[2])
        return LoadTask(device, profile, level)
    else:
        raise ValueError("Invalid format. Use 'device:random' or 'device:profile:level'")

def create_generator(device: str) -> LoadGenerator:
    if device == "cpu":
        return CPULoadGenerator()
    elif device == "gpu":
        if GPULoadGenerator:
            return GPULoadGenerator()
        else:
            print("[Error] GPU support not available (PyTorch not found).")
            return None
    elif device == "npu":
        print("[Warn] NPU not implemented in Python version yet.")
        return None
    else:
        raise ValueError(f"Device not supported: {device}")

def main():
    global stop_requested
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <device:profile:level> ...")
        sys.exit(1)

    generators: List[LoadGenerator] = []
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        for arg in sys.argv[1:]:
            task = parse_arg(arg)
            gen = create_generator(task.device)
            if gen:
                # 优化显示逻辑：Random 模式显示为 Dynamic
                level_display = "Dynamic (Wave)" if task.profile == ProfileType.Random else task.level.name
                print(f"Starting {task.device} | Profile: {task.profile.name} | Level: {level_display}")
                
                gen.start(task.profile, task.level)
                generators.append(gen)
    except Exception as e:
        print(f"Init Error: {e}")
        for g in generators: g.stop()
        sys.exit(1)

    print("Running... Press Ctrl+C to stop.")
    
    try:
        while not stop_requested:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping all generators...")
        for g in generators:
            g.stop()
        print("Done.")

if __name__ == "__main__":
    # multiprocessing.freeze_support()
    main()