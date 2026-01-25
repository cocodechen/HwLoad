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
    void runCompute();
    void runMemory();
    void runData();
#ifdef __linux__
    void runIO();
#endif
    void memory_worker();
    void disk_worker();
    void cpu_core_worker(int core_id);
    void runRandom();

    std::atomic<bool> running;
    std::thread worker;
};

#endif