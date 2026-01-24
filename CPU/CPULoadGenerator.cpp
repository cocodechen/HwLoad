#ifdef HWLOAD_USE_CPU

#include "CPULoadGenerator.hpp"
#include <cmath>
#include <vector>
#include <thread>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>

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

    if (profile == ProfileType::Random) {
        worker = std::thread(&CPULoadGenerator::runRandom, this);
    }
    else if (profile == ProfileType::Compute) {
        worker = std::thread(&CPULoadGenerator::runCompute, this);
    }
    else if (profile == ProfileType::Memory) {
        worker = std::thread(&CPULoadGenerator::runMemory, this);
    }
    else if (profile == ProfileType::Data) {
        worker = std::thread(&CPULoadGenerator::runData, this);
    }
#ifdef __linux__
    else if (profile == ProfileType::IO) {
        worker = std::thread(&CPULoadGenerator::runIO, this);
    }
#endif
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

void CPULoadGenerator::runCompute()
{
    unsigned num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> pool;

    for (unsigned t = 0; t < num_threads; ++t) {
        pool.emplace_back([&]() {
            while (running) {
                auto [loops, sleep_ms] = get_cpu_params(cur_level);
                if (loops > 0) do_cpu_burn(loops);
                if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            }
        });
    }
    for (auto& t : pool) t.join();
}

void CPULoadGenerator::runMemory()
{
    unsigned num_threads = std::thread::hardware_concurrency();
    const size_t TOTAL_SIZE = 256 * 1024 * 1024; 
    const size_t CHUNK_SIZE = TOTAL_SIZE / (num_threads ? num_threads : 1);

    std::vector<char> buffer(TOTAL_SIZE, 0);
    std::vector<std::thread> pool;

    for (unsigned t = 0; t < num_threads; ++t) {
        pool.emplace_back([&, t]() {
            size_t start = t * CHUNK_SIZE;
            size_t end = start + CHUNK_SIZE;
            while (running) {
                auto [loops, sleep_ms] = get_cpu_params(cur_level);
                // 内存通常比计算快，loop减半防止太卡
                if (loops > 0) do_memory_burn(buffer, start, end, std::max(1, loops/2));
                if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            }
        });
    }
    for (auto& t : pool) t.join();
}

void CPULoadGenerator::runData()
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

    for (unsigned t = 0; t < num_threads; ++t) {
        pool.emplace_back([&, t]() {
            // 每个线程从不同的起点开始跑，减少 False Sharing
            int cur_idx = indices[(t * 1000) % NUM_ELEMENTS];
            
            while (running)
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
void CPULoadGenerator::runIO()
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

    while (running) {
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

// --- 单个 CPU 核心的工作逻辑 ---
void CPULoadGenerator::cpu_core_worker(int core_id)
{
    double offset = static_cast<double>(core_id * 10);
    
    // 使用 std::chrono 进行精确的时间片控制
    using namespace std::chrono;
    const int TIME_SLICE_MS = 100; // 每个周期总长 100ms

    // 预分配计算资源，避免反复 new/delete 消耗系统时间而非 CPU 时间
    const int MAT_SIZE = 1024;
    std::vector<double> data_a(MAT_SIZE, 1.0);
    std::vector<double> data_b(MAT_SIZE, 2.0);

    while (running) {
        auto cycle_start = high_resolution_clock::now();
        double target_load = get_wave_intensity(offset); // 0.0 - 1.0

        // 设定本周期的工作时长
        double work_ms = target_load * TIME_SLICE_MS;
        double sleep_ms = (1.0 - target_load) * TIME_SLICE_MS;

        // --- 阶段1: 暴力计算 (Burn CPU) ---
        // 持续运算直到达到 work_ms 时长
        // 这种方式能保证无论 CPU 多快，占用率都是准确的
        if (work_ms > 0.5) { // 至少工作 0.5ms
            while (running) {
                // 检查是否超时
                auto now = high_resolution_clock::now();
                duration<double, std::milli> elapsed = now - cycle_start;
                if (elapsed.count() >= work_ms) break;

                // 密集浮点运算 (简单的乘加运算，模拟 FPU 压力)
                // 使用 volatile 防止编译器优化掉循环
                volatile double val = 0;
                for (int i = 0; i < 1000; ++i) {
                    val += data_a[i] * data_b[i] + 0.001;
                }
                (void)val;
            }
        }

        // --- 阶段2: 偶尔的内存/IO 压力 (降低频率) ---
        // 只有在负载很高时，且每 10 个周期(约1秒)触发一次，避免拖累 CPU%
        static thread_local int counter = 0;
        if (target_load > 0.7 && ++counter % 10 == 0) {
            // 简单的内存带宽压力：快速写入
            // 注意：这里不做分配，只做写操作
            std::fill(data_a.begin(), data_a.end(), target_load);
        }

        // --- 阶段3: 精确休眠 ---
        if (sleep_ms > 1.0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<long>(sleep_ms)));
        } else {
            // 极高负载时，让出时间片即可，不深度睡眠
            std::this_thread::yield();
        }
    }
}

void CPULoadGenerator::runRandom()
{
    // 1. 获取硬件并发数
    unsigned int num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 4; // fallback

    std::cout << "[CPU Manager] Detected " << num_cores << " cores. Spawning workers..." << std::endl;

    // 2. 启动 Worker 线程池
    std::vector<std::thread> cpu_workers;
    cpu_workers.reserve(num_cores);

    for (unsigned int i = 0; i < num_cores; ++i) {
        // 将外部的 running 引用传递给子线程
        cpu_workers.emplace_back(&CPULoadGenerator::cpu_core_worker, this, i);
    }

    // 3. 等待外部信号停止
    // 主控线程在这里阻塞，直到所有子线程结束
    for (auto& t : cpu_workers) {
        if (t.joinable()) {
            t.join();
        }
    }

    std::cout << "[CPU Manager] All CPU workers stopped." << std::endl;
}

#endif