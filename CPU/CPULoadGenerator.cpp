#ifdef HWLOAD_USE_CPU

#include "CPULoadGenerator.hpp"
#include "CPUReal.cpp"

#include <cmath>
#include <vector>
#include <thread>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#ifdef __linux__
    #include <unistd.h>
    #include <fcntl.h>
    #include <sys/mman.h>   
#endif

namespace fs = std::filesystem;

CPULoadGenerator::CPULoadGenerator(): running(false){}

CPULoadGenerator::~CPULoadGenerator() {
    stop();
}

void CPULoadGenerator::start(ProfileType profile, LoadLevel level)
{
    cur_level = level;
    running = true;
    if (profile == ProfileType::Compute) {
        worker = std::thread(&CPULoadGenerator::runCompute, this, std::chrono::milliseconds(0));
    }
    else if (profile == ProfileType::Memory) {
        worker = std::thread(&CPULoadGenerator::runMemory, this, std::chrono::milliseconds(0));
    }
    else if (profile == ProfileType::Data) {
        worker = std::thread(&CPULoadGenerator::runData, this, std::chrono::milliseconds(0));
    }
#ifdef __linux__
    else if (profile == ProfileType::IO) {
        worker = std::thread(&CPULoadGenerator::runIO, this, std::chrono::milliseconds(0));
    }
#endif
    else if (profile == ProfileType::Random) {
        worker = std::thread(&CPULoadGenerator::runRandom, this);
    }
    else if (profile == ProfileType::Real) {
        worker = std::thread(&CPULoadGenerator::runReal, this);
    }
    else throw std::runtime_error("[CPULoad] No support profile");
}

void CPULoadGenerator::stop()
{
    running.store(false);
    if (worker.joinable())worker.join();
}

// --- 内部辅助：防止优化 ---
static void do_cpu_burn(long loops) {
    volatile double val = 1.0; 
    for (long i = 0; i < loops * 1000; ++i) {
        val = val * 1.0000001 + 0.0001;
    }
}

// --- 内部辅助：内存 ---
static void do_memory_burn(std::vector<char>& buffer, size_t start, size_t end, long loops) {
    // 步长 64 (Cache Line)
    const size_t STRIDE = 64;
    volatile char* ptr = buffer.data(); // volatile防止循环被优化移除
    for (int l = 0; l < loops; ++l) {
        for (size_t i = start; i < end; i += STRIDE) {
            ptr[i] = (char)l; // 单纯写，带宽压力更大
        }
    }
}

static std::pair<int, int> get_cpu_params(LoadLevel level)
{
    switch (level)
    {
        case LoadLevel::Idle:      return {1, 500};
        case LoadLevel::Low:       return {100, 20};
        case LoadLevel::Medium:    return {500, 5};
        case LoadLevel::High:      return {1000, 0};
        case LoadLevel::Saturated: return {5000, 0};
        default: return {100, 10};
    }
}

bool CPULoadGenerator::shouldContinue(std::chrono::steady_clock::time_point start_time, std::chrono::milliseconds duration)
{
    // 1. 如果全局开关已关闭，立即停止
    if (!running) return false;

    // 2. 如果 duration 为 0，表示无限运行模式，不检查时间
    if (duration.count() <= 0) return true;

    // 3. 检查是否超时
    auto now = std::chrono::steady_clock::now();
    return (now - start_time) < duration;
}

void CPULoadGenerator::runCompute(std::chrono::milliseconds duration)
{
    unsigned num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> pool;

    // 记录开始时间
    auto start_time = std::chrono::steady_clock::now();

    for (unsigned t = 0; t < num_threads; ++t) {
        pool.emplace_back([&]() {
            while (shouldContinue(start_time, duration)) {
                auto [loops, sleep_ms] = get_cpu_params(cur_level);
                if (loops > 0) do_cpu_burn(loops);
                if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            }
        });
    }
    for (auto& t : pool) t.join();
}

void CPULoadGenerator::runMemory(std::chrono::milliseconds duration)
{
    unsigned num_threads = std::thread::hardware_concurrency();
    const size_t TOTAL_SIZE = 256 * 1024 * 1024; 
    const size_t CHUNK_SIZE = TOTAL_SIZE / (num_threads ? num_threads : 1);

    std::vector<char> buffer(TOTAL_SIZE, 0);
    std::vector<std::thread> pool;

    auto start_time = std::chrono::steady_clock::now(); // 记录时间

    for (unsigned t = 0; t < num_threads; ++t) {
        pool.emplace_back([&, t]() {
            size_t start = t * CHUNK_SIZE;
            size_t end = start + CHUNK_SIZE;
            while (shouldContinue(start_time, duration)) {
                auto [loops, sleep_ms] = get_cpu_params(cur_level);
                // 内存通常比计算快，loop减半防止太卡
                if (loops > 0) do_memory_burn(buffer, start, end, std::max(1, loops/2));
                if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            }
        });
    }
    for (auto& t : pool) t.join();
}

void CPULoadGenerator::runData(std::chrono::milliseconds duration)
{
    // Data Profile: 混合计算与访存，使用指针追逐 (Pointer Chasing)
    // 这是一个单线程或少线程的测试，主要看延迟
    // 为了制造负载，我们还是开多线程
    unsigned num_threads = std::thread::hardware_concurrency();
    
    // 数据集大小：64MB (足够大，击穿 L3)
    const size_t NUM_ELEMENTS = 16 * 1024 * 1024; // 16M ints * 4 = 64MB
    
    struct Node {
        int next;
        int padding[15]; // 64 bytes total (cache line)
    };
    
    std::vector<Node> data(NUM_ELEMENTS);
    
    // 初始化随机链表 (Fisher-Yates shuffle)
    std::vector<int> indices(NUM_ELEMENTS);
    for(size_t i=0; i<NUM_ELEMENTS; ++i) indices[i] = i;
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    for(size_t i=0; i<NUM_ELEMENTS - 1; ++i) {
        data[indices[i]].next = indices[i+1];
    }
    data[indices[NUM_ELEMENTS-1]].next = indices[0]; // 闭环

    std::vector<std::thread> pool;
    auto start_time = std::chrono::steady_clock::now(); // 记录时间

    for (unsigned t = 0; t < num_threads; ++t) {
        pool.emplace_back([&, t]() {
            // 每个线程从不同的起点开始跑，减少 False Sharing
            int cur_idx = indices[(t * 1000) % NUM_ELEMENTS];
            
            while (shouldContinue(start_time, duration))
            {
                auto [loops, sleep_ms] = get_cpu_params(cur_level);
                if (loops == 0)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
                    continue;
                }
                // 指针追逐核心
                for (int i = 0; i < loops * 1000; ++i)
                {
                    cur_idx = data[cur_idx].next;
                    // 稍微加一点计算，模拟混合负载
                    data[cur_idx].padding[0]++; 
                }

                if (sleep_ms > 0)
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            }
        });
    }

    for (auto& t : pool) t.join();
}

#ifdef __linux__
void CPULoadGenerator::runIO(std::chrono::milliseconds duration)
{
    // 保持你原有的 IO 逻辑，稍微调整大小和 Sleep
    const size_t map_size = 256 * 1024 * 1024; // 固定 256MB，避免 Level 间差异过大

    int fd = open("/tmp/cpu_io_tmp.bin", O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd < 0) return;

    if (ftruncate(fd, map_size) != 0) { close(fd); return; }

    char* p = static_cast<char*>(
        mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    
    if (p == MAP_FAILED) { close(fd); return; }

    const size_t stride = 4096; // Page size
    auto start_time = std::chrono::steady_clock::now(); // 记录时间

    while (shouldContinue(start_time, duration)) {
        auto [loops, sleep_ms] = get_cpu_params(cur_level);
        if (loops == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            continue;
        }

        int io_loops = std::max(1, loops / 50); 
        int io_sleep = std::max(10, sleep_ms); // IO 需要更多等待防止系统卡死
        
        // 模拟脏页回写压力
        for (int l = 0; l < loops / 10 + 1; ++l) { // IO 很慢，loop 缩小
            for (size_t i = 0; i < map_size; i += stride) {
                p[i]++; 
            }
            // 异步刷盘，触发内核 IO 线程负载
            msync(p, map_size, MS_ASYNC);
        }

        if (sleep_ms > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }

    munmap(p, map_size);
    close(fd);
    unlink("/tmp/cpu_io_tmp.bin");
}
#endif

void CPULoadGenerator::runRandom()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<ProfileType> profiles = {
        ProfileType::Compute, 
        ProfileType::Memory, 
        ProfileType::Data
    };
#ifdef __linux__
    profiles.push_back(ProfileType::IO);
#endif

    std::vector<LoadLevel> levels = {
        LoadLevel::Idle, LoadLevel::Low, LoadLevel::Medium, 
        LoadLevel::High, LoadLevel::Saturated
    };

    while (running) // 这里只检查全局开关，因为 runRandom 本身就是无限循环直到停止
    {
        // 1. 生成随机参数
        std::uniform_int_distribution<> dur_dist(10000, 20000); // 10~20秒
        std::uniform_int_distribution<> prof_dist(0, profiles.size() - 1);
        std::uniform_int_distribution<> lvl_dist(0, levels.size() - 1);

        auto duration = std::chrono::milliseconds(dur_dist(gen));
        ProfileType p = profiles[prof_dist(gen)];

        std::cout
            << "[LoadGen_CPU] "
            << "Profile=" << profile2Str(p)
            << ", Level=" << level2Str(cur_level)
            << ", Duration=" << duration.count() << " ms ("
            << duration.count() / 1000.0 << " s)"
            << std::endl;
        
        // 设置新的负载等级
        cur_level = levels[lvl_dist(gen)];

        // 2. 调用已有的功能函数 (传入时长)
        // 它们会运行指定时长后自动返回
        if (p == ProfileType::Compute) {
            runCompute(duration);
        }
        else if (p == ProfileType::Memory) {
            runMemory(duration);
        }
        else if (p == ProfileType::Data) {
            runData(duration);
        }
#ifdef __linux__
        else if (p == ProfileType::IO) {
            runIO(duration);
        }
#endif
        // 3. 简短休眠，让 CPU 喘口气，也防止切换过于剧烈
        if (running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

void CPULoadGenerator::runReal()
{
    execute_real_simulation(this->running);
}

#endif