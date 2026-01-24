#ifdef HWLOAD_USE_GPU

#include "GPULoadGenerator.hpp"
#include <stdexcept>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <algorithm>

// 宏定义... (同前)
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "[GPULoad] CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return; \
    } \
} while (0)

__global__ void k_compute(float* data, int intensity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = idx * 0.0001f;
    #pragma unroll
    for (int i = 0; i < intensity; ++i) {
        val = val * 1.000001f + __sinf(val) * 0.000001f;
    }
    data[idx] = val;
}

__global__ void k_memory(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = src[i] + 1.0f; 
    }
}

struct GPUConfig
{
    int blocks;
    int threads;
    int kernel_loops; // 一个 launch 里的计算强度
    int sleep_ms;     // launch 之间的间隔
};

static GPUConfig get_gpu_config(LoadLevel level, ProfileType type)
{
    // 默认配置 (Medium/High/Saturated 基准)
    int threads = 256; 
    int blocks = 2048; 

    int k_loops = 0;
    int sleep = 0;

    switch (level) {
        case LoadLevel::Idle:
            // 心跳模式：最小 Kernel，长睡眠
            // 1 Block, 32 Threads (1 Warp) 是最小执行粒度
            blocks = 1; threads = 32; k_loops = 1; sleep = 500; 
            break;
        case LoadLevel::Low:       
            k_loops = 500; sleep = 30; 
            break;
        case LoadLevel::Medium:    
            k_loops = 2000; sleep = 10; 
            break;
        case LoadLevel::High:      
            k_loops = 10000; sleep = 0; 
            break;
        case LoadLevel::Saturated: 
            // 饱和模式：撑满 GPU 调度队列
            blocks = 8192; k_loops = 50000; sleep = 0; 
            break;
    }

    // Memory Profile 特殊处理
    if (type == ProfileType::Memory) {
        // Memory 主要靠数据量(blocks数量)，k_loops 影响重复拷贝次数
        if (level == LoadLevel::Idle) {
            k_loops = 1; // 拷一次就行
        } else if (level == LoadLevel::Saturated) {
            k_loops = 50;
        } else {
            k_loops = 10;
        }
    }

    return {blocks, threads, k_loops, sleep};
}

// 构造函数
GPULoadGenerator::GPULoadGenerator() { running.store(false); }

GPULoadGenerator::~GPULoadGenerator(){
    stop();
}

void GPULoadGenerator::start(ProfileType profile, LoadLevel level)
{
    cur_level = level;
    running.store(true);

    if (profile == ProfileType::Compute) {
        worker = std::thread(&GPULoadGenerator::runCompute, this);
    }
    else if (profile == ProfileType::Memory) {
        worker = std::thread(&GPULoadGenerator::runMemory, this);
    }
    else if(profile == ProfileType::Data){
        worker = std::thread(&GPULoadGenerator::runData, this);
    }
    else if(profile == ProfileType::Random){
        worker = std::thread(&GPULoadGenerator::runRandom, this);
    }
    else throw std::runtime_error("[GPULoad] No support profile");
}

void GPULoadGenerator::stop()
{
    running.store(false);
    if (worker.joinable()) worker.join();
}

void GPULoadGenerator::runCompute()
{
    CUDA_CHECK(cudaSetDevice(0));
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, 1024 * 1024 * 32 * sizeof(float)));
    auto cfg = get_gpu_config(cur_level, ProfileType::Compute);

    while (running.load(std::memory_order_relaxed))
    {
        k_compute<<<cfg.blocks, cfg.threads>>>(d_data, cfg.kernel_loops); 
        if (cur_level != LoadLevel::Saturated)cudaDeviceSynchronize();
        if (cfg.sleep_ms > 0)std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
    }
    CUDA_CHECK(cudaFree(d_data));
}

void GPULoadGenerator::runMemory()
{
    CUDA_CHECK(cudaSetDevice(0));

    // 数据量必须大，击穿 L2 Cache (通常 40-80MB)
    // 128MB floats = 512MB 显存占用 x 2 (src+dst) = 1GB
    // 这个大小对大多数显卡安全，且足够击穿 Cache
    const int N = 1024 * 1024 * 128; 
    float *d_src = nullptr, *d_dst = nullptr;

    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
    
    // 初始化一下数据，避免 page fault 干扰（虽然 GPU 上影响较小）
    cudaMemset(d_src, 0, N * sizeof(float));

    while (running) {
        auto cfg = get_gpu_config(cur_level, ProfileType::Memory);
        
        // 调整 block 数量以覆盖数据
        int min_blocks = (N + cfg.threads - 1) / cfg.threads;
        int actual_blocks = std::max(cfg.blocks, min_blocks);

        if (cur_level != LoadLevel::Idle) {
            // 反复拷贝
            for(int i=0; i< std::max(1, cfg.kernel_loops / 10); ++i) {
                k_memory<<<actual_blocks, cfg.threads>>>(d_dst, d_src, N);
            }
        }

        cudaDeviceSynchronize(); // Memory 测试必须 Sync，否则测不出带宽瓶颈

        if (cfg.sleep_ms > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
    }

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

void GPULoadGenerator::runData()
{
    // Data Profile: 混合 Compute 和 Memory
    CUDA_CHECK(cudaSetDevice(0));

    const int N = 1024 * 1024 * 64; // 64M floats = 256MB
    float *d_src = nullptr, *d_dst = nullptr;

    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));

    while (running) {
        auto cfg = get_gpu_config(cur_level, ProfileType::Data); // 这里的 loop 参数偏向 Compute
        
        int min_blocks = (N + cfg.threads - 1) / cfg.threads;

        if (cur_level != LoadLevel::Idle) {
            // 1. 先做计算
            k_compute<<<cfg.blocks, cfg.threads>>>(d_src, cfg.kernel_loops);
            
            // 2. 再做搬运 (数据依赖)
            k_memory<<<min_blocks, cfg.threads>>>(d_dst, d_src, N);
        }
        
        cudaDeviceSynchronize();

        if (cfg.sleep_ms > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
    }

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

// --- CUDA Kernel ---
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

// --- 单个 GPU 的工作逻辑 ---
void GPULoadGenerator::gpu_device_worker(int device_id)
{
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // 预分配 Host 内存
    const int MAX_N = 4096;
    std::vector<float> host_data(MAX_N * MAX_N, 1.0f);

    while (running) {
        double target_load = get_wave_intensity(device_id * 20.0);

        if (target_load < 0.1) {
            // 清理缓存并休眠
            cudaDeviceSynchronize();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }

        // 计算矩阵大小 (1000 ~ 4000)
        int N = 1000 + static_cast<int>(3000 * target_load);
        if (N > MAX_N) N = MAX_N;
        
        size_t bytes = N * N * sizeof(float);
        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

        // 分配显存 (模拟显存波动)
        if (cudaMalloc(&d_A, bytes) != cudaSuccess) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); continue; }
        if (cudaMalloc(&d_B, bytes) != cudaSuccess) { cudaFree(d_A); continue; }
        if (cudaMalloc(&d_C, bytes) != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); continue; }

        // 拷贝数据 (模拟 PCIe 带宽)
        cudaMemcpy(d_A, host_data.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, host_data.data(), bytes, cudaMemcpyHostToDevice);

        // Kernel 配置
        dim3 threads(16, 16);
        dim3 blocks((N + 15) / 16, (N + 15) / 16);

        // 执行计算
        // 循环次数随负载增加
        int loops = std::max(1, static_cast<int>(10 * target_load));
        for (int i = 0; i < loops && running; ++i) {
            matrixMulKernel<<<blocks, threads>>>(d_C, d_A, d_B, N);
        }
        
        // 同步
        cudaDeviceSynchronize();

        // 释放显存
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // 动态休眠
        int sleep_ms = static_cast<int>((1.0 - target_load) * 100);
        if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }
}

// --- GPU 管理器函数 ---
// 这个函数会被外部的一个线程调用。
// 它负责检测 GPU 数量并为每个 GPU 启动子线程
void GPULoadGenerator::runRandom()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "[GPU Manager] No CUDA devices found or CUDA not installed." << std::endl;
        return;
    }

    std::cout << "[GPU Manager] Detected " << deviceCount << " GPU(s). Spawning workers..." << std::endl;

    std::vector<std::thread> workers;
    workers.reserve(deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        workers.emplace_back(&GPULoadGenerator::gpu_device_worker, this, i);
    }

    // 等待所有 GPU 线程结束
    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }

    std::cout << "[GPU Manager] All GPU workers stopped." << std::endl;
}

#endif