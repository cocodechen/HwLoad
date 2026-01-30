# HwLoad - Multi-Architecture Hardware Load Generator

**HwLoad** is a lightweight, high-performance, and cross-platform hardware load generator. It is designed to stress-test specific hardware components (CPU, GPU, NPU) by generating precise synthetic loads or simulating realistic workloads.

It supports granular control over load intensity (from idle keep-alive to saturation) and allows **mixed-device testing** (e.g., stressing CPU and GPU simultaneously).

---

## ✨ Key Features

*   **Multi-Architecture**: Native support for **CPU** (x86/ARM), **NVIDIA GPU** (CUDA), and **Ascend NPU** (MindSpore).
*   **Granular Control**: 5 distinct pressure levels ranging from "Idle" to "Saturated".
*   **Targeted Profiles**: Isolate specific bottlenecks (ALU, Memory Bandwidth, Latency, IO).
*   **Chaos Mode**: "Random" profile to simulate fluctuating cluster environments.
*   **Real Simulation**: "Real" profile to mimic actual model training/inference loops.
*   **Concurrency**: Run multiple load generators in parallel via a single command.

---

## 🛠️ Installation & Build

### Prerequisites
| Component | Requirement | Note |
| :--- | :--- | :--- |
| **Compiler** | C++17 compliant | GCC, Clang, or MSVC |
| **CMake** | ≥ 3.10 | Build system |
| **CUDA** | Toolkit 11.0+ | Required for `HWLOAD_USE_GPU` |
| **MindSpore** | Python Environment | Required for `HWLOAD_USE_NPU` |

### Compilation
Configure the build to enable specific hardware backends:

```bash
mkdir build && cd build
# Enable/Disable modules as needed
cmake .. \
    -DHWLOAD_USE_CPU=ON \
    -DHWLOAD_USE_GPU=ON \
    -DHWLOAD_USE_NPU=ON 

make -j
```


## 🚀 Usage

The binary accepts a list of tasks in the format device:profile:level. You can combine multiple tasks to stress different hardware simultaneously.

```bash
./loadgen <device:profile:level> [device:profile:level] ...
```

## 📊 Profiles & Internals

### 1. CPU (Host)
- **Implementation**: C++11 threads and OS primitives  
- **Compute**: Complex floating-point math (`sin`, `sqrt`, arithmetic). Uses `std::atomic` and `volatile` to prevent compiler optimization  
- **Memory**: Allocates a buffer larger than L3 cache (256 MB). Performs linear and strided writes to saturate DRAM bandwidth  
- **Data**: Pointer chasing. Traverses a randomized linked list (64 MB) to maximize cache misses and stress memory latency  
- **IO (Linux only)**: Uses `mmap` to map a file, writes dirty pages, and forces disk synchronization via `msync`
- **Random**: Randomly select a profile and level.
- **Real**
---

### 2. GPU (NVIDIA CUDA)
- **Implementation**: CUDA Runtime API  
- **Compute**: Fused Multiply-Add (FMA) and transcendental functions in tight loops to saturate SMs (Streaming Multiprocessors)  
- **Memory**: Device-to-device (D2D) memory copies and read–modify–write operations to saturate HBM/GDDR bandwidth  
- **Data**: Mixed kernel with compute-dependent memory access, simulating general-purpose CUDA workloads
- **Random**: Randomly select a profile and level.
- **Real**
---

### 3. NPU (Huawei Ascend)
- **Implementation**: MindSpore (Python bridge)  
- **Compute**: Large `MatMul` (FP16) operators to saturate Cube Units  
- **Memory**: `Add` / `Assign` (FP32) on large tensors to stress Vector Units and HBM bandwidth 
- **Data**: Mixed FP32↔FP16 casting with FP16 MatMul to stress data movement, Vector Units, and memory bandwidth.
- **Random**: Randomly select a profile and level.
- **Real**: Simulates a CNN training step (`Conv2d + BN + ReLU`). Uses internal repeats to keep the NPU busy without CPU/Python overhead

---

## 🎚️ Load Levels

Levels determine the duty cycle (run/sleep ratio), thread count, and data size.

| Level | Description | Behavior |
|------|-------------|----------|
| Idle | Keep-Alive | Minimal pulse (e.g., 1 ms work / 500 ms sleep). Keeps the context active for monitoring |
| Low | Background | Short bursts with long sleep intervals (e.g., 20–30 ms sleep) |
| Medium | Business | Balanced work/sleep ratio. Simulates typical application usage |
| High | Heavy | Continuous operation with minimal sleep. Targets >90% utilization |
| Saturated | Stress / Max | 0 ms sleep + max concurrency. Spawns threads = logical cores (CPU) or fills command queues (GPU/NPU). Tests thermal limits and power throttling |

---

## ⚠️ Notes
- **Privileges**: `cpu:io` may require root privileges depending on the write location  
- **Heat generation**: `Saturated` level can generate significant heat. Ensure adequate cooling before long runs  
