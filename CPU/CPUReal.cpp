#include <thread>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <atomic>

namespace fs = std::filesystem;

// --- 辅助函数：获取模拟波形负载强度 (0.0 - 1.0) ---
// --- 0.5单峰 ---
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

// 左右双峰
// inline double get_wave_intensity(double offset_seed) {
//     auto now = std::chrono::system_clock::now();
//     double t = std::chrono::duration<double>(now.time_since_epoch()).count();

//     // =========================================================
//     // 1. 宏观趋势 (Macro Trend) - 决定“忙闲阶段”
//     // =========================================================
//     // 周期：约 60秒 (t / 10.0)
//     // 作用：让基准线在 [0.2, 0.8] 之间缓慢游走。
//     // 当它为正时，整体处于“高负载区”；为负时，处于“低负载区”。
//     double macro_trend = 0.35 * std::sin(t / 10.0 + offset_seed * 0.5);
    
//     // 动态基准线：不再固定是 0.5，而是随时间变化
//     double dynamic_base = 0.5 + macro_trend; 

//     // =========================================================
//     // 2. 活跃波动 (Micro Fluctuation) - 模拟“正在处理任务”
//     // =========================================================
//     // 周期：约 3秒，频率较快
//     // 幅度：0.15，让曲线在基准线上下跳动，但不至于由于幅度太大而一直被截断
//     double active_wave = 0.15 * std::sin(t / 0.5 + offset_seed * 2.0);

//     // =========================================================
//     // 3. 随机突发/噪声 (Noise & Spikes)
//     // =========================================================
//     static thread_local std::mt19937 gen(std::random_device{}());
//     std::uniform_real_distribution<> noise_dist(-0.05, 0.05);
    
//     // 偶尔加入一个“突发任务” (Spike)
//     // 利用 fmod 模拟每 13 秒左右可能出现一次短暂的脉冲
//     double spike = 0.0;
//     // 使用 t 的小数部分或其他周期函数来触发
//     if (std::sin(t / 2.0 + offset_seed) > 0.95) { 
//         spike = 0.3; // 突然增加 30% 负载
//     }

//     // =========================================================
//     // 4. 合成
//     // =========================================================
//     // 结果 = 动态基准 + 活跃波动 + 噪声 + 突发
//     double val = dynamic_base + active_wave + noise_dist(gen) + spike;

//     // 钳制
//     return std::max(0.01, std::min(val, 1.0));
// }

// ============================================================
// 1. CPU Worker: 纯计算线程
// ============================================================

void cpu_core_worker(int core_id,std::atomic<bool>& running)
{
    // ==========================================
    // 【参数调优区】 - 修改这里来控制负载行为
    // ==========================================
    
    // 1. 负载区间映射
    // 如果觉得整体偏低，把 MIN 调高 (比如 0.50)，MAX 调成 1.0
    const double MIN_TARGET_LOAD = 0.20; 
    const double MAX_TARGET_LOAD = 0.95; 

    // 2. 刷新窗口 (100ms 通常是比较平衡的值)
    const int WINDOW_MS = 100; 

    // 3. 计算粒度 (关键优化变量)
    // 这个值越大，查时间的频率越低，利用率越高。
    // 如果你的 CPU 很快，建议设为 50000 或 100000。
    // 如果太小（比如 1000），CPU 都在忙着获取系统时间，利用率上不去。
    const int loops = 10; 

    // 4. 忙等待阈值 (单位：微秒)
    // 如果剩余休眠时间小于这个值，就不睡了，直接空转死循环。
    // Windows 建议 16000 (16ms)，Linux 建议 2000 (2ms)。
    // 设得越大，CPU 占用越高（因为这部分时间被算作工作了）。
#ifdef __linux__
    const long long SPIN_THRESHOLD_US = 2000; 
#else
    const long long SPIN_THRESHOLD_US = 16000; 
#endif
    // ==========================================

    std::mt19937 gen(std::random_device{}() + core_id);
    std::uniform_real_distribution<> dis(0.0, 10.0);
    double offset = dis(gen);

    // 防优化变量
    volatile double dummy = 0.0; 

    while (running) {
        auto cycle_start = std::chrono::high_resolution_clock::now();

        // 计算目标负载
        double wave = get_wave_intensity(offset); 
        // 映射波形到 [MIN, MAX]
        double target_load = MIN_TARGET_LOAD + wave * (MAX_TARGET_LOAD - MIN_TARGET_LOAD);
        target_load = std::clamp(target_load, 0.0, 1.0);

        long long work_ns = static_cast<long long>(target_load * WINDOW_MS * 1000000);

        // --- 暴力计算阶段 ---
        while (running) {
            auto now = std::chrono::high_resolution_clock::now();
            if ((now - cycle_start).count() >= work_ns) {
                break;
            }

             // 执行一小段计算 (比如耗时 1微秒)
            // 这样能频繁检查时间，保证精度
            volatile double val = 1.0; 
            for (long i = 0; i < loops * 1000; ++i) {
                val = val * 1.0000001 + 0.0001;
            }
        }

        if (!running) break;

        // --- 修正休眠阶段 ---
        auto work_end = std::chrono::high_resolution_clock::now();
        long long elapsed = (work_end - cycle_start).count();
        long long total_window_ns = WINDOW_MS * 1000000;
        long long remaining_ns = total_window_ns - elapsed;

        if (remaining_ns > 0) {
            // 如果剩余时间小于阈值，直接忙等待 (视为 100% 负载)
            if (remaining_ns < SPIN_THRESHOLD_US * 1000) { 
                auto spin_start = std::chrono::high_resolution_clock::now();
                while ((std::chrono::high_resolution_clock::now() - spin_start).count() < remaining_ns) {
                    // 空转，死死占住 CPU
                }
            } else {
                // 时间充裕，真正睡眠 (CPU 占用率为 0%)
                std::this_thread::sleep_for(std::chrono::nanoseconds(remaining_ns));
            }
        }
    }
}

// ============================================================
// 2. Memory Worker: 独立内存线程 
// ============================================================
void memory_worker(std::atomic<bool>& running)
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
void disk_worker(std::atomic<bool>& running)
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
// 启动真实负载模拟
// 参数 running: 引用主类的控制开关，用于通知子线程停止
void execute_real_simulation(std::atomic<bool>& running)
{
    unsigned int num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 4;

    std::cout << "[CPUReal] Starting " << num_cores << " compute threads, 1 Memory thread, 1 Disk thread..." << std::endl;

    std::vector<std::thread> threads;

    // 1. 启动 CPU 线程
    for (unsigned int i = 0; i < num_cores; ++i) {
        // 使用 lambda 包装调用，传递 running 引用
        threads.emplace_back([i, &running]() {
            cpu_core_worker(i, running);
        });
    }

    // 2. 启动 Memory 线程
    threads.emplace_back([&running]() {
        memory_worker(running);
    });

    // 3. 启动 Disk 线程
    threads.emplace_back([&running]() {
        disk_worker(running);
    });

    // 等待所有线程结束
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
    
    std::cout << "[CPUReal] All CPU workers stopped." << std::endl;
}
