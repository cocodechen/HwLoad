#ifdef HWLOAD_USE_GPU

#include "GPULoadGenerator.hpp"
#include "GPUReal.cu"

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
        worker = std::thread(&GPULoadGenerator::runCompute, this, std::chrono::milliseconds(0));
    }
    else if (profile == ProfileType::Memory) {
        worker = std::thread(&GPULoadGenerator::runMemory, this, std::chrono::milliseconds(0));
    }
    else if(profile == ProfileType::Data){
        worker = std::thread(&GPULoadGenerator::runData, this, std::chrono::milliseconds(0));
    }
    else if(profile == ProfileType::Random){
        worker = std::thread(&GPULoadGenerator::runRandom, this);
    }
    else if(profile==ProfileType::Real){
        worker = std::thread(&GPULoadGenerator::runReal, this);
    }
    else throw std::runtime_error("[GPULoad] No support profile");
}

void GPULoadGenerator::stop()
{
    running.store(false);
    if (worker.joinable()) worker.join();
}

bool GPULoadGenerator::shouldContinue(std::chrono::steady_clock::time_point start_time, std::chrono::milliseconds duration)
{
    // 1. 检查全局开关 (使用 relaxed 即可，性能敏感度低)
    if (!running.load(std::memory_order_relaxed)) return false;

    // 2. duration <= 0 代表无限运行
    if (duration.count() <= 0) return true;

    // 3. 检查是否超时
    return (std::chrono::steady_clock::now() - start_time) < duration;
}

void GPULoadGenerator::runCompute(std::chrono::milliseconds duration)
{
    CUDA_CHECK(cudaSetDevice(0));
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, 1024 * 1024 * 32 * sizeof(float)));

    // 记录开始时间
    auto start_time = std::chrono::steady_clock::now();
    
    while (shouldContinue(start_time, duration))
    {
        // 获取配置 (每次循环都获取，允许外部动态调整 level，虽然这里 level 由 random 控制)
        auto cfg = get_gpu_config(cur_level, ProfileType::Compute);

        k_compute<<<cfg.blocks, cfg.threads>>>(d_data, cfg.kernel_loops); 
        if (cur_level != LoadLevel::Saturated)cudaDeviceSynchronize();
        if (cfg.sleep_ms > 0)std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms));
    }
    CUDA_CHECK(cudaFree(d_data));
}

void GPULoadGenerator::runMemory(std::chrono::milliseconds duration)
{
    CUDA_CHECK(cudaSetDevice(0));
    const int N = 1024 * 1024 * 128; 
    float *d_src = nullptr, *d_dst = nullptr;

    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
    cudaMemset(d_src, 0, N * sizeof(float));

    auto start_time = std::chrono::steady_clock::now();

    while (shouldContinue(start_time, duration)) {
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

void GPULoadGenerator::runData(std::chrono::milliseconds duration)
{
    // Data Profile: 混合 Compute 和 Memory
    CUDA_CHECK(cudaSetDevice(0));
    const int N = 1024 * 1024 * 64; // 64M floats = 256MB
    float *d_src = nullptr, *d_dst = nullptr;

    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));

    auto start_time = std::chrono::steady_clock::now();

    while (shouldContinue(start_time, duration))
    {
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

void GPULoadGenerator::runRandom()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<ProfileType> profiles = {
        ProfileType::Compute, 
        ProfileType::Memory, 
        ProfileType::Data
    };

    std::vector<LoadLevel> levels = {
        LoadLevel::Idle, LoadLevel::Low, LoadLevel::Medium, 
        LoadLevel::High, LoadLevel::Saturated
    };

    while (running)
    {
        // 1. 随机生成持续时间 (10s - 20s)
        std::uniform_int_distribution<> dur_dist(10000, 20000); 
        auto duration = std::chrono::milliseconds(dur_dist(gen));

        // 2. 随机选择 Profile 和 Level
        std::uniform_int_distribution<> prof_dist(0, profiles.size() - 1);
        std::uniform_int_distribution<> lvl_dist(0, levels.size() - 1);

        ProfileType p = profiles[prof_dist(gen)];
        cur_level = levels[lvl_dist(gen)]; // 更新当前 Level，get_gpu_config 会读取它

        std::cout
            << "[LoadGen_GPU] "
            << "Profile=" << profile2Str(p)
            << ", Level=" << level2Str(cur_level)
            << ", Duration=" << duration.count() << " ms ("
            << duration.count() / 1000.0 << " s)"
            << std::endl;

        // 3. 执行对应的负载函数 (运行指定时长)
        if (p == ProfileType::Compute) {
            runCompute(duration);
        }
        else if (p == ProfileType::Memory) {
            runMemory(duration);
        }
        else if (p == ProfileType::Data) {
            runData(duration);
        }

        // 4. 短暂休眠，模拟任务切换间隙
        if (running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

void GPULoadGenerator::runReal()
{
    // 调用外部文件的函数，传入当前的运行状态标志位
    // 这里的 this->running 是 std::atomic<bool> 类型
    execute_real_gpu_simulation(this->running);
}

#endif