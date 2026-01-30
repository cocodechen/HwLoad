#include "LoadGenerator.hpp"
#ifdef HWLOAD_USE_CPU
    #include "CPU/CPULoadGenerator.hpp"
#endif
#ifdef HWLOAD_USE_GPU
    #include "GPU/GPULoadGenerator.hpp"
#endif
#ifdef HWLOAD_USE_NPU
    #include "NPU/NPULoadGenerator.hpp"
#endif

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <sstream>

std::atomic<bool> g_stop_requested{false};

static void signal_handler(int signal) {
    if (signal == SIGINT) g_stop_requested.store(true);
}


std::unique_ptr<LoadGenerator> createGenerator(const std::string& device) {
    if (device == "cpu") {
#ifdef HWLOAD_USE_CPU
        return std::make_unique<CPULoadGenerator>();
#endif
    } else if (device == "gpu") {
#ifdef HWLOAD_USE_GPU
        return std::make_unique<GPULoadGenerator>();
#endif
    } else if (device == "npu") {
#ifdef HWLOAD_USE_NPU
        return std::make_unique<NPULoadGenerator>();
#endif
    }
    throw std::runtime_error("Device not supported or enabled: " + device);
}

// 解析格式: "cpu:compute:high"
LoadTask parseArg(const std::string& arg)
{
    std::stringstream ss(arg);
    std::string segment;
    std::vector<std::string> parts;
    while(std::getline(ss, segment, ':')) {
        parts.push_back(segment);
    }
    // 格式 1: device:random (parts=2)
    if (parts.size() == 2) {
        ProfileType p = str2Profile(parts[1]);
        if (p == ProfileType::Random || p == ProfileType::Real)
        {
            // Random/Real 不需要 Level，给一个默认值即可
            return {parts[0], p, LoadLevel::Medium}; 
        } else {
            throw std::invalid_argument("Profile '" + parts[1] + "' requires a level (e.g., :high)");
        }
    }
    // 格式 2: device:compute:high (parts=3)
    else if (parts.size() == 3) {
        ProfileType p = str2Profile(parts[1]);
        if (p == ProfileType::Random) {
             // 如果用户非要写 cpu:random:high，也可以兼容，或者报错
             // 这里选择兼容
             return {parts[0], p, LoadLevel::Medium};
        }
        return {parts[0], p, str2Level(parts[2])};
    }

    throw std::invalid_argument("Invalid format. Use 'device:random' or 'device:profile:level'");
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: loadgen <device:profile:level> [device:profile:level] ...\n"
                  << "Example: loadgen cpu:compute:high gpu:random\n";
        return 1;
    }

    std::cout << "Running... Press Ctrl+C to stop.\n";
    std::signal(SIGINT, signal_handler);

    std::vector<std::unique_ptr<LoadGenerator>> generators;
    try
    {
        for (int i = 1; i < argc; ++i) {
            auto task = parseArg(argv[i]);
            auto gen = createGenerator(task.device);
            // 优化启动顺序：先放入 vector，再 start
            // 这样即使 start 抛出异常，vector 析构时也能正确释放 gen
            // 前提是 LoadGenerator 的析构函数必须安全 (见下文)
            generators.push_back(std::move(gen));
            generators.back()->start(task.profile, task.level);
            std::cout << "Started " << task.device << " (" << argv[i] << ")\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Init Error: " << e.what() << "\n";
        // 发生错误时，停止已启动的
        for (auto& g : generators) if(g) g->stop();
        return 1;
    }


    while (!g_stop_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Stopping all generators...\n";
    for (auto& g : generators) {
        if(g) g->stop();
    }
    
    return 0;
}