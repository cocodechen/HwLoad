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
    void runCompute();
    void runMemory();
    void runData();

    void gpu_device_worker(int device_id);
    void runRandom();

    std::atomic<bool> running;
    std::thread worker;
};

#endif