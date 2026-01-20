#ifdef HWLOAD_USE_GPU

#include "GPULoadGenerator.hpp"

#include <stdexcept>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error: "                            \
                  << cudaGetErrorString(err)                  \
                  << " at " << __FILE__ << ":" << __LINE__    \
                  << std::endl;                                \
        std::abort();                                         \
    }                                                         \
} while (0)

inline void CUDA_CHECK_LAUNCH(const char* stage)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA launch error at " << stage << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::abort();
    }
}

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
        throw std::runtime_error("GPU is not supported on Profile IO");
    }
}

void GPULoadGenerator::stop()
{
    running = false;
    if (worker.joinable())worker.join();
}

__global__ void gpu_compute_kernel(float* data, int iters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = idx * 0.001f;

    #pragma unroll 4
    for (int i = 0; i < iters * 1000; ++i) {   // ★ 放大计算强度
        x = x * 1.0000001f + 0.0000001f;
    }

    if (idx < (1 << 20))
        data[idx] = x;   // ★ 防止编译器优化掉
}

__global__ void gpu_memory_kernel(float* dst, float* src)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx];
}

static inline std::chrono::milliseconds level_to_time(LoadLevel lv)
{
    switch (lv) {
        case LoadLevel::Idle:      return std::chrono::milliseconds(200);
        case LoadLevel::Low:       return std::chrono::milliseconds(50);
        case LoadLevel::Medium:    return std::chrono::milliseconds(10);
        case LoadLevel::High:      return std::chrono::milliseconds(1);
        case LoadLevel::Saturated: return std::chrono::milliseconds(0);
        default:                   return std::chrono::milliseconds(20);
    }
}

static inline int level_to_iters(LoadLevel lv)
{
    switch (lv) {
        case LoadLevel::Idle:      return 1;
        case LoadLevel::Low:       return 2;
        case LoadLevel::Medium:    return 6;
        case LoadLevel::High:      return 16;
        case LoadLevel::Saturated: return 32;
        default:                   return 6;
    }
}

void GPULoadGenerator::runCompute()
{
    CUDA_CHECK(cudaSetDevice(0));

    constexpr int N = 1 << 20;
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));

    int blocks  = 256;
    int threads = 256;

    int iters = level_to_iters(cur_level);
    auto idle = level_to_time(cur_level);

    while (running)
    {
        if (cur_level != LoadLevel::Idle) {
            gpu_compute_kernel<<<blocks, threads>>>(d_data, iters);
            CUDA_CHECK_LAUNCH("gpu_compute_kernel");
        }

        if (idle.count() > 0)
            std::this_thread::sleep_for(idle);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_data));
}

void GPULoadGenerator::runMemory()
{
    CUDA_CHECK(cudaSetDevice(0));

    constexpr int N = 1 << 22;
    float *src = nullptr, *dst = nullptr;

    CUDA_CHECK(cudaMalloc(&src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst, N * sizeof(float)));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    auto idle = level_to_time(cur_level);

    while (running)
    {
        if (cur_level != LoadLevel::Idle) {
            gpu_memory_kernel<<<blocks, threads>>>(dst, src);
            CUDA_CHECK_LAUNCH("gpu_memory_kernel");
        }

        if (idle.count() > 0)
            std::this_thread::sleep_for(idle);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
}

void GPULoadGenerator::runData()
{
    CUDA_CHECK(cudaSetDevice(0));

    constexpr int N = 1 << 22;
    float *src = nullptr, *dst = nullptr;

    CUDA_CHECK(cudaMalloc(&src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst, N * sizeof(float)));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    int iters = level_to_iters(cur_level);
    auto idle = level_to_time(cur_level);

    while (running)
    {
        if (cur_level != LoadLevel::Idle) {
            gpu_compute_kernel<<<blocks, threads>>>(dst, iters);
            CUDA_CHECK_LAUNCH("gpu_compute_kernel");

            gpu_memory_kernel<<<blocks, threads>>>(dst, src);
            CUDA_CHECK_LAUNCH("gpu_memory_kernel");
        }

        if (idle.count() > 0)
            std::this_thread::sleep_for(idle);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
}


#endif