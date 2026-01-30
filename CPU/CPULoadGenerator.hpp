#pragma once

#ifdef HWLOAD_USE_CPU

#include "LoadGenerator.hpp"

#include <atomic>
#include <thread>

class CPULoadGenerator : public LoadGenerator
{
public:
    CPULoadGenerator();
    ~CPULoadGenerator();
    
    void start(ProfileType profile,LoadLevel level) override;
    void stop() override;

private:
    void runCompute(std::chrono::milliseconds duration = std::chrono::milliseconds(0));
    void runMemory(std::chrono::milliseconds duration = std::chrono::milliseconds(0));
    void runData(std::chrono::milliseconds duration = std::chrono::milliseconds(0));
#ifdef __linux__
    void runIO(std::chrono::milliseconds duration = std::chrono::milliseconds(0));
#endif
    // 统一判断是否继续运行的辅助函数
    bool shouldContinue(std::chrono::steady_clock::time_point start_time, std::chrono::milliseconds duration);
    void runRandom();

    void runReal();

    std::atomic<bool> running;
    std::thread worker;
};

#endif