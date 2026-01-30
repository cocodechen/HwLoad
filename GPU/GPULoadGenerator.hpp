#pragma once

#ifdef HWLOAD_USE_GPU

#include "LoadGenerator.hpp"

#include <atomic>
#include <thread>

class GPULoadGenerator : public LoadGenerator
{
public:
    GPULoadGenerator();
    ~GPULoadGenerator();

    void start(ProfileType profile,LoadLevel level) override;
    void stop() override;

private:
    void runCompute(std::chrono::milliseconds duration = std::chrono::milliseconds(0));
    void runMemory(std::chrono::milliseconds duration = std::chrono::milliseconds(0));
    void runData(std::chrono::milliseconds duration = std::chrono::milliseconds(0));

    // 辅助函数
    bool shouldContinue(std::chrono::steady_clock::time_point start_time, std::chrono::milliseconds duration);
    void runRandom();

    void runReal();

    std::atomic<bool> running;
    std::thread worker;
};

#endif