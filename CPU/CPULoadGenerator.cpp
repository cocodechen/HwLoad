#ifdef HWLOAD_USE_CPU

#include "CPULoadGenerator.hpp"
#include <chrono>
#include <cmath>
#include <vector>
#include <atomic>
#include <thread>
#include <algorithm>
#include <random>
#include <cstring> // for memset

#ifdef __linux__
    #include <unistd.h>
    #include <fcntl.h>
    #include <sys/mman.h>   
    #include <sys/stat.h>
#endif

CPULoadGenerator::CPULoadGenerator(): running(false){}

void CPULoadGenerator::start(ProfileType profile, LoadLevel level)
{
    cur_level = level;
    running = true;
    
    // 统一在这里创建线程，避免在 runXXX 内部重复逻辑
    // 根据 Profile 选择具体的 worker 函数
    if (profile == ProfileType::Compute) {
        worker = std::thread(&CPULoadGenerator::runCompute, this);
    }
    else if (profile == ProfileType::Memory) {
        worker = std::thread(&CPULoadGenerator::runMemory, this);
    }
    else if (profile == ProfileType::Data) {
        worker = std::thread(&CPULoadGenerator::runData, this);
    }
    else if (profile == ProfileType::IO)
    {
#ifdef __linux__
        worker = std::thread(&CPULoadGenerator::runIO, this);
#else
        throw std::runtime_error("CPU IO load only supported on Linux");
#endif
    }
}

void CPULoadGenerator::stop()
{
    running = false;
    if (worker.joinable()) worker.join();
}

// 辅助：获取不同 Level 的休眠时间 (Duty Cycle 控制)
// 返回值: (work_loops, sleep_ms)
static std::pair<int, int> get_cpu_duty_cycle(LoadLevel level, ProfileType type) {
    // 基础参数
    int sleep_ms = 0;
    int loops = 1;

    switch (level) {
        case LoadLevel::Idle:      
            // 心跳模式：极微小的负载，长睡眠。证明线程活着，但几乎不占 CPU
            sleep_ms = 500; loops = 1; break; 
            
        case LoadLevel::Low:       
            sleep_ms = 20;  loops = 100;  break; // ~20% load
            
        case LoadLevel::Medium:    
            sleep_ms = 5;   loops = 500;  break; // ~60% load
            
        case LoadLevel::High:      
            sleep_ms = 0;   loops = 1000; break; // ~100% load
            
        case LoadLevel::Saturated: 
            sleep_ms = 0;   loops = 5000; break; // Max
    }
    
    // 针对 Memory Profile 微调
    // 内存操作比纯计算慢得多，Idle 时保持 loops=1 测一下延迟即可，不需要减半
    if (type == ProfileType::Memory && level != LoadLevel::Idle) {
        loops /= 2; 
    }
    
    return {loops, sleep_ms};
}

void CPULoadGenerator::runCompute()
{
    unsigned num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 2;

    std::vector<std::thread> pool;
    std::atomic<uint64_t> global_sink{0};

    for (unsigned t = 0; t < num_threads; ++t) {
        pool.emplace_back([&]() {
            while (running) {
                auto [loops, sleep_ms] = get_cpu_duty_cycle(cur_level, ProfileType::Compute);

                if (loops == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
                    continue;
                }

                // 计算核心
                double x = 1.0;
                for (int i = 0; i < loops * 1000; ++i) {
                    x = x * 1.0000001 + std::sin(x) * 0.001;
                    // 简单的分支，增加流水线压力
                    if (x > 1000.0) x = 1.0;
                }
                
                // 防止优化
                global_sink.fetch_add(static_cast<uint64_t>(x), std::memory_order_relaxed);

                if (sleep_ms > 0)
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            }
        });
    }

    for (auto& t : pool) t.join();
}

void CPULoadGenerator::runMemory()
{
    unsigned num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 2;

    // 固定 buffer 大小：256MB
    // 这足以击穿绝大多数 CPU 的 L3 Cache (通常 20-64MB)
    // 又不至于让小内存机器 OOM
    const size_t TOTAL_SIZE = 256 * 1024 * 1024;
    const size_t CHUNK_SIZE = TOTAL_SIZE / num_threads;

    std::vector<char> buffer(TOTAL_SIZE, 0);
    std::vector<std::thread> pool;

    for (unsigned t = 0; t < num_threads; ++t) {
        pool.emplace_back([&, t]() {
            size_t start = t * CHUNK_SIZE;
            size_t end = (t == num_threads - 1) ? TOTAL_SIZE : start + CHUNK_SIZE;
            
            // 步长设置为 64 字节 (Typical Cache Line Size)
            // 保证每次访问都命中不同的 Cache Line
            const size_t STRIDE = 64; 

            while (running) {
                auto [loops, sleep_ms] = get_cpu_duty_cycle(cur_level, ProfileType::Memory);

                if (loops == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
                    continue;
                }

                for (int l = 0; l < loops; ++l) {
                    // 顺序写：对预取器友好，主要测带宽上限
                    for (size_t i = start; i < end; i += STRIDE) {
                        buffer[i]++;
                    }
                }
                
                // 防止 buffer 被优化掉（虽然有副作用操作通常不会）
                std::atomic_thread_fence(std::memory_order_release);

                if (sleep_ms > 0)
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
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
            
            while (running) {
                auto [loops, sleep_ms] = get_cpu_duty_cycle(cur_level, ProfileType::Data);

                if (loops == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
                    continue;
                }

                // 指针追逐核心
                for (int i = 0; i < loops * 1000; ++i) {
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
        auto [loops, sleep_ms] = get_cpu_duty_cycle(cur_level, ProfileType::IO);
        if (loops == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            continue;
        }

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

#endif