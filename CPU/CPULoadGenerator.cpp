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


// ============================================================
// 1. CPU Worker: 纯计算线程
// ============================================================

void CPULoadGenerator::cpu_core_worker(int core_id)
{
    // ==========================================
    // 核心参数：控制波动的上下限
    const double MIN_TARGET_LOAD = 0.20; // 20%
    const double MAX_TARGET_LOAD = 0.95; // 95%
    
    // 时间片窗口：100ms 是任务管理器刷新的合适粒度
    const int WINDOW_MS = 100; 
    // ==========================================

    std::mt19937 gen(std::random_device{}() + core_id);
    std::uniform_real_distribution<> dis(0.0, 10.0);
    double offset = dis(gen);

    while (running) {
        // 1. 标记周期开始时刻
        auto cycle_start = std::chrono::high_resolution_clock::now();

        // 2. 获取原始波形 (0.0 - 1.0)
        double wave = get_wave_intensity(offset);

        // 3. 映射到目标区间 [20%, 95%]
        // Python 没做这个映射，是因为 Python 本身跑得慢，天然就把负载拉高了
        // C++ 必须手动控制区间，否则很容易掉到底
        double target_load = MIN_TARGET_LOAD + wave * (MAX_TARGET_LOAD - MIN_TARGET_LOAD);
        target_load = std::clamp(target_load, 0.0, 1.0);

        // 4. 计算应该工作多久 (Work Duration)
        long long work_ns = static_cast<long long>(target_load * WINDOW_MS * 1000000);
        long long sleep_ns = static_cast<long long>((1.0 - target_load) * WINDOW_MS * 1000000);

        // 5. --- 暴力计算阶段 (强制工作 work_ns 纳秒) ---
        while (running) {
            auto now = std::chrono::high_resolution_clock::now();
            if ((now - cycle_start).count() >= work_ns) {
                break; // 时间到了，去休息
            }

            // 执行一小段计算 (比如耗时 1微秒)
            // 这样能频繁检查时间，保证精度
            do_cpu_burn(1); 
        }

        if (!running) break;

        // 6. --- 修正休眠阶段 ---
        // 重新计算实际消耗的时间，用剩下的时间去睡
        auto work_end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(work_end - cycle_start).count();
        long long remaining_ns = (WINDOW_MS * 1000000) - elapsed;

        if (remaining_ns > 0) {
            // 如果剩余时间很少 (<2ms)，也就是负载很高(>90%)时
            // 不要用 sleep，因为 sleep 可能会睡过头导致掉负载
            if (remaining_ns < 2000000) { 
                // 忙等待 (Spin Wait) - 虽然是等待，但在任务管理器看来是 100% 占用
                // 但为了让出一点点给系统，用 yield
                auto sleep_start = std::chrono::high_resolution_clock::now();
                while ((std::chrono::high_resolution_clock::now() - sleep_start).count() < remaining_ns) {
                    std::this_thread::yield();
                }
            } else {
                // 时间充裕，正常睡眠，节省功耗
                std::this_thread::sleep_for(std::chrono::nanoseconds(remaining_ns));
            }
        }
    }
}
// ============================================================
// 2. Memory Worker: 独立内存线程 
// ============================================================
void CPULoadGenerator::memory_worker()
{
    // 模拟 Python: allocated_data = []
    // vector of vector 用来模拟 list of bytearray
    std::vector<std::vector<char>> allocated_data;
    
    // 为了防止 C++ bad_alloc 崩溃，设置一个物理上限 (比如 64GB)
    // Python 脚本里 target_gb = 100 * target_load，可能很大
    const size_t MAX_SAFE_GB = 64; 
    const size_t CHUNK_SIZE = 100 * 1024 * 1024; // 100MB

    while (running) {
        try {
            double target_load = get_wave_intensity(20.0); // Offset 20

            // Python: target_gb = 100 * target_load
            double target_gb = 100.0 * target_load;
            
            // 限制最大值
            if (target_gb > MAX_SAFE_GB) target_gb = MAX_SAFE_GB;

            // Python: current_held_gb = len(allocated_data) * 0.1
            double current_held_gb = allocated_data.size() * 0.1;

            if (current_held_gb < target_gb) {
                // Python: allocated_data.append(bytearray(os.urandom(chunk_size)))
                // C++: 分配并填充
                try {
                    std::vector<char> chunk(CHUNK_SIZE);
                    // 关键：必须写入数据，否则 OS 可能会延迟分配物理内存 (Lazy Allocation)
                    // 使用 memset 模拟 os.urandom 的部分开销 (写内存)
                    std::memset(chunk.data(), 1, CHUNK_SIZE); 
                    allocated_data.push_back(std::move(chunk));
                } catch (const std::bad_alloc&) {
                    // 忽略内存不足，休息一下
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            } 
            else if (current_held_gb > target_gb + 2.0) { // +2 GB 滞后区间
                // Python: allocated_data.pop()
                if (!allocated_data.empty()) {
                    allocated_data.pop_back();
                }
            }

            // Python: time.sleep(0.5)
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

        } catch (...) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    // 线程退出时 allocated_data 会自动释放
}

// ============================================================
// 3. Disk Worker: 独立IO线程 
// ============================================================
void CPULoadGenerator::disk_worker()
{
    // Python: disk_file_path = ...
    fs::path temp_dir = fs::temp_directory_path();
    fs::path disk_file_path = temp_dir / "stress_test_file.dat";
    
    // buffer
    const size_t BLOCK_SIZE = 1024 * 1024; // 1MB
    std::vector<char> data(BLOCK_SIZE, 'A'); // 模拟 random data

    while (running) {
        double target_load = get_wave_intensity(40.0); // Offset 40

        if (target_load < 0.1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        // Python: file_size_mb = int(10 + 490 * target_load)
        int file_size_mb = static_cast<int>(10 + 490 * target_load);

        try {
            // --- 写 (Write) ---
            {
                std::ofstream ofs(disk_file_path, std::ios::binary | std::ios::trunc);
                if (ofs) {
                    for (int i = 0; i < file_size_mb; ++i) {
                        if (!running) break;
                        ofs.write(data.data(), BLOCK_SIZE);
                    }
                }
            }
            if (!running) break;

            // --- 读 (Read) ---
            if (fs::exists(disk_file_path)) {
                std::ifstream ifs(disk_file_path, std::ios::binary);
                if (ifs) {
                    std::vector<char> read_buf(BLOCK_SIZE);
                    while (ifs.read(read_buf.data(), BLOCK_SIZE)) {
                        if (!running) break;
                    }
                }
            }

            // --- 删 (Remove) ---
            if (fs::exists(disk_file_path)) {
                fs::remove(disk_file_path);
            }

        } catch (...) {
            // ignore errors
        }

        // Python: time.sleep((1.0 - target_load) * 1.0)
        if (running) {
            double sleep_sec = (1.0 - target_load) * 1.0;
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<long>(sleep_sec * 1000)));
        }
    }

    // 清理
    if (fs::exists(disk_file_path)) {
        try { fs::remove(disk_file_path); } catch(...) {}
    }
}

// ============================================================
// 4. Manager: 启动所有线程
// ============================================================
void CPULoadGenerator::runRandom()
{
    unsigned int num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 4;

    std::cout << "[CPU Manager] Starting " << num_cores << " compute threads, 1 Memory thread, 1 Disk thread..." << std::endl;

    std::vector<std::thread> threads;

    // 1. 启动 CPU 计算线程 (1 Core = 1 Thread)
    for (unsigned int i = 0; i < num_cores; ++i) {
        threads.emplace_back(&CPULoadGenerator::cpu_core_worker, this, i);
    }

    // 2. 启动 内存 线程
    threads.emplace_back(&CPULoadGenerator::memory_worker, this);

    // 3. 启动 磁盘 线程
    threads.emplace_back(&CPULoadGenerator::disk_worker, this);

    // 等待所有线程结束 (由外部调用 stop() 触发 running=false)
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    std::cout << "[CPU Manager] All workers stopped." << std::endl;
}

#endif