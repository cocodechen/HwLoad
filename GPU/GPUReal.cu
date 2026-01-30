#include <cuda_runtime.h>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <mutex>

// 简单的宏定义，避免依赖外部头文件
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "[RealGpuLoad] CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    } \
} while (0)
#endif

// ==========================================
// 内部辅助函数 (Static)
// ==========================================

static double get_wave_intensity(double offset_seed)
{
    auto now = std::chrono::system_clock::now();
    // 获取秒数
    double t = std::chrono::duration<double>(now.time_since_epoch()).count();

    // 1. 基础趋势波 (周期很长，约 40秒)
    // 范围 [-0.5, 0.5]
    double slow_wave = 0.5 * std::sin(t / 20.0 + offset_seed);

    // 2. 活跃波动波 (周期较短，约 5秒)
    // 范围 [-0.2, 0.2]
    double fast_wave = 0.2 * std::sin(t / 2.5 + offset_seed * 2.0);

    // 3. 随机抖动 (让曲线看起来不那么平滑)
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> noise_dist(-0.05, 0.05);
    double noise = noise_dist(gen);

    // 叠加: 基准值 0.5 + 慢波 + 快波 + 噪声
    // 结果大概率落在 [0.1, 0.9] 之间，偶尔触顶或触底
    double val = 0.5 + slow_wave + fast_wave + noise;

    // 钳制在 [0.0, 1.0] 安全范围
    return std::max(0.01, std::min(val, 1.0));
}
// ==========================================
// CUDA Kernel
// ==========================================

// 简单的矩阵乘法核心，用于产生 GPU 热量
static __global__ void matrixMulKernel(float* C, const float* A, const float* B, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ==========================================
// Worker 实现
// ==========================================

static void gpu_device_worker(int device_id, std::atomic<bool>& running)
{
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // 获取显存总量
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    // 预分配 Host 内存 (加大到 8192*8192，约 256MB)
    const int MAX_N = 8192;
    std::vector<float> host_data(MAX_N * MAX_N);
    // 填充一点数据，避免全是0
    std::fill(host_data.begin(), host_data.begin() + 10240, 1.23f);

    // 时间片 200ms
    const int TIME_SLICE_MS = 200;

    // 显存占位指针
    void* d_vram_holder = nullptr;
    size_t current_holder_size = 0;

    while (running) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        double target_load = get_wave_intensity(device_id * 10.0);

        // ============================
        // 1. 显存波动控制 (VRAM)
        // ============================
        
        // 重新获取一下当前 Free
        cudaMemGetInfo(&free_mem, &total_mem);
        
        // 计算我们想要占用的显存量 (target_load * 可用显存)
        // 预留 500MB 给矩阵计算用
        size_t available_for_grab = (free_mem + current_holder_size > 500 * 1024 * 1024) 
                                    ? (free_mem + current_holder_size - 500 * 1024 * 1024) 
                                    : 0;
        
        size_t target_size = static_cast<size_t>(available_for_grab * target_load);

        // 【防抖动】只有当目标大小和当前大小差异超过 100MB 时，才进行重新分配
        if (std::abs((long long)target_size - (long long)current_holder_size) > 100 * 1024 * 1024) {
            // 释放旧的
            if (d_vram_holder) {
                cudaFree(d_vram_holder);
                d_vram_holder = nullptr;
                current_holder_size = 0;
            }
            // 分配新的
            if (target_size > 0) {
                if (cudaMalloc(&d_vram_holder, target_size) == cudaSuccess) {
                    current_holder_size = target_size;
                    // 可选：在高负载时 async memset 一小部分
                    if (target_load > 0.7) {
                        cudaMemsetAsync(d_vram_holder, 0, std::min(target_size, (size_t)10 * 1024 * 1024), 0);
                    }
                }
            }
        }

        // ============================
        // 2. 计算波动控制 (Compute)
        // ============================
        
        long work_ms = static_cast<long>(target_load * TIME_SLICE_MS);
        
        // 如果需要工作
        if (work_ms > 5) {
            // 矩阵大小固定大一点，或者随负载微调
            int N = 2048 + static_cast<int>((MAX_N - 2048) * target_load);
            size_t bytes = N * N * sizeof(float);
            
            float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
            
            bool success = true;
            if (cudaMalloc(&d_A, bytes) != cudaSuccess) success = false;
            if (success && cudaMalloc(&d_B, bytes) != cudaSuccess) success = false;
            if (success && cudaMalloc(&d_C, bytes) != cudaSuccess) success = false;

            if (success) {
                // 拷贝 (PCIe 负载)
                cudaMemcpyAsync(d_A, host_data.data(), bytes, cudaMemcpyHostToDevice, 0);
                cudaMemcpyAsync(d_B, host_data.data(), bytes, cudaMemcpyHostToDevice, 0);
                
                dim3 threads(16, 16);
                dim3 blocks((N + 15) / 16, (N + 15) / 16);

                while (running) {
                    matrixMulKernel<<<blocks, threads, 0, 0>>>(d_C, d_A, d_B, N);
                    cudaStreamSynchronize(0);

                    auto now = std::chrono::high_resolution_clock::now();
                    long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                    if (elapsed >= work_ms) break;
                }
            }
            
            if (d_A) cudaFree(d_A);
            if (d_B) cudaFree(d_B);
            if (d_C) cudaFree(d_C);
        }

        // ============================
        // 3. 休息
        // ============================
        auto end_work_time = std::chrono::high_resolution_clock::now();
        long actual_work_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_work_time - start_time).count();
        long sleep_ms = TIME_SLICE_MS - actual_work_ms;

        if (sleep_ms > 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
    }

    if (d_vram_holder) cudaFree(d_vram_holder);
}

// ==========================================
// 主入口函数
// ==========================================

void execute_real_gpu_simulation(std::atomic<bool>& running)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "[RealGpuLoad] No CUDA devices found or CUDA not installed." << std::endl;
        return;
    }

    std::cout << "[RealGpuLoad] Detected " << deviceCount << " GPU(s). Spawning real-world simulation workers..." << std::endl;

    std::vector<std::thread> workers;
    workers.reserve(deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        workers.emplace_back([i, &running]() {
            gpu_device_worker(i, running);
        });
    }

    // 等待所有 GPU 线程结束
    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }

    std::cout << "[RealGpuLoad] All GPU workers stopped." << std::endl;
}