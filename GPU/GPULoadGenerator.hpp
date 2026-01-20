#pragma once

#ifdef HWLOAD_USE_GPU

#include "LoadGenerator.hpp"

#include <atomic>
#include <thread>

class GPULoadGenerator : public LoadGenerator
{
public:
    GPULoadGenerator();

    void start(ProfileType profile,LoadLevel level) override;
    void stop() override;

private:
    void runCompute();
    void runMemory();
    void runData();

    std::atomic<bool> running;
    std::thread worker;
};

#endif