#ifdef HWLOAD_USE_GPU

#include "GPULoadGenerator.hpp"
#include <stdexcept>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <algorithm>

// 简化的错误检查宏
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "[GPULoad] CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return; /* 发生错误直接返回，避免 abort 导致整个程序崩溃 */ \
    } \
} while (0)

GPULoadGenerator::GPULoadGenerator(): running(false){}

void GPULoadGenerator::start(ProfileType profile, LoadLevel level)
{
    cur_level = level;
    running = true;

    if (profile == ProfileType::Compute) {
        worker = std::thread(&GPULoadGenerator::runCompute, this);
    }
    else if (profile == ProfileType::Memory) {
        worker = std::thread(&GPULoadGenerator::runMemory, this);
    }
    else if(profile == ProfileType::Data){
        worker = std::thread(&GPULoadGenerator::runData, this);
    }
    else if(profile == ProfileType::IO){
        // GPU IO 一般指 GPUDirectStorage，这里暂不支持，打印警告即可
        throw std::runtime_error("[GPULoad] Warning: GPU IO profile not supported yet.");
    }
}

void GPULoadGenerator::stop()
{
    running = false;
    if (worker.joinable()) worker.join();
}

// ==========================================
// Kernels
// ==========================================

// 计算密集型 Kernel
// 增加循环次数，确保 SM 忙碌
__global__ void k_compute(float* data, int intensity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = idx * 0.0001f;

    // 避免编译器优化掉循环
    #pragma unroll 4
    for (int i = 0; i < intensity; ++i) {
        val = val * 1.000001f + 0.000001f;
        // 添加一些超越函数计算，利用 SFU (Special Function Unit)
        if (i % 100 == 0) val = __sinf(val); 
    }
    
    data[idx] = val;
}

// 访存密集型 Kernel
// 简单的 Copy，主要看带宽
__global__ void k_memory(float* dst, const float* src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx] + 1.0f; // Read + Write
    }
}

// ==========================================
// Helpers
// ==========================================

struct GPUConfig {
    int blocks;
    int threads;
    int kernel_loops; // 一个 launch 里的计算强度
    int sleep_ms;     // launch 之间的间隔
};

static GPUConfig get_gpu_config(LoadLevel level, ProfileType type) {
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

// ==========================================
// Implementations
// ==========================================

void GPULoadGenerator::runCompute()
{
    CUDA_CHECK(cudaSetDevice(0));

    // 只需要少量的显存来存结果
    const int N = 1024 * 1024 * 32; // 32M floats = 128MB
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));

    while (running) {
        auto cfg = get_gpu_config(cur_level, ProfileType::Compute);

        if (cur_level != LoadLevel::Idle) {
            // Compute Kernel
            k_compute<<<cfg.blocks, cfg.threads>>>(d_data, cfg.kernel_loops);
            // 异步 Launch，不需要立即 Sync
        }

        // 只有 Saturated 不 Sync，保持队列充满
        if (cur_level != LoadLevel::Saturated) {
            cudaDeviceSynchronize();
        }

        if (cfg.sleep_ms > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
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

#endif