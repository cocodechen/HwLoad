# HwLoad - Multi-Architecture Hardware Load Generator

**HwLoad** 是一个轻量级、跨平台的硬件负载生成器。它能够针对不同的硬件架构（CPU, GPU, NPU）生成特定类型的负载（计算密集、访存密集、数据延迟、IO），并支持从空闲心跳到饱和压力的多档位调节。

---

## 支持的硬件与负载类型 (Profiles)

### 1. CPU (Host)
基于 C++11 多线程与系统调用实现。
- **Compute (计算密集)**: 执行复杂的浮点运算（`sin`, `log`, `sqrt`），利用 `std::atomic` 防止编译器优化，主要向 ALU 施压。
- **Memory (访存密集)**: 分配超过 L3 Cache 的大内存块（固定 256MB），进行顺序与跨步读写，测试 DRAM 带宽。
- **Data (数据/延迟)**: 指针追逐（Pointer Chasing）算法，构建随机链表遍历，制造大量 Cache Miss，测试内存延迟与流水线停顿。
- **IO (磁盘/页缓存)**: Linux 专用。使用 `mmap` 映射文件并定期执行 `msync`，模拟脏页回写压力。

### 2. GPU (NVIDIA CUDA)
基于 CUDA Runtime API 实现。
- **Compute**: 启动高并发 Kernel 进行 FP32 乘加运算与超越函数计算，旨在打满 SM (Streaming Multiprocessor)。
- **Memory**: 在显存（HBM/GDDR）间进行大规模数据拷贝（D2D Copy），击穿 L2 Cache，测试显存带宽。
- **Data**: 混合 Kernel，结合了计算与依赖数据的搬运，模拟通用的数据处理流水线。

### 3. NPU (Huawei Ascend)
基于 MindSpore (Python) 调用底层算子。
- **Compute**: 使用 `MatMul` (FP16) 算子，利用 **Cube Unit** 进行矩阵乘法，算力利用率极高。
- **Memory**: 使用 `Add/Mul` + `Assign` (FP32) 算子，强制数据回写 HBM，利用 **Vector Unit** 并压榨显存带宽。
- **Data**: 混合精度计算，包含 `Cast` (类型转换) 和 `MatMul`，模拟实际训练中的混合精度场景。

---

## 压力档位 (Levels)

通过调节 **Duty Cycle (占空比)**、**并发量** 和 **数据规模** 来控制负载强度。

| Level | 描述 | 典型行为 |
| :--- | :--- | :--- |
| **Idle** | **心跳/保活** | 极低频运行 (如每 500ms 跑一次微小负载)。保持 Context 活跃，便于监控系统确认进程存活，几乎无功耗。 |
| **Low** | **低负载** | 短时间运行，长时间休眠 (Sleep 20~30ms)。模拟轻量级后台任务。 |
| **Medium** | **中负载** | 运行与休眠时间相当。模拟正常业务负载。 |
| **High** | **高负载** | 几乎不休眠，连续运行。旨在将利用率推至 90% 以上。 |
| **Saturated**| **饱和/烤机** | **不休眠 + 最大化并发**。CPU 开启最大线程数，GPU/NPU 启动最大 Grid/Matrix。旨在测试散热极限、最大功耗或触发硬件 Throttling。 |

---

## 🚀 编译与使用

### 前置要求
*   **Compiler**: GCC/Clang (支持 C++11)
*   **CUDA**: NVIDIA CUDA Toolkit (用于 GPU 负载)
*   **MindSpore**: 昇思环境 (用于 NPU 负载，需 Python 环境)
*   **Build**: CMake 3.10+

### 编译构建
```bash
mkdir build && cd build
cmake ..
make
```

### 运行命令
```bash
./HwLoad --device <cpu|gpu|npu> --profile <type> --level <level>
```