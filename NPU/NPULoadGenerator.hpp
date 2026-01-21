#pragma once

#ifdef HWLOAD_USE_NPU

#include "LoadGenerator.hpp"
#include <unistd.h> // for pid_t

class NPULoadGenerator : public LoadGenerator
{
public:
    NPULoadGenerator();
    ~NPULoadGenerator(); // 析构时确保子进程被清理

    void start(ProfileType profile, LoadLevel level) override;
    void stop() override;

private:
    pid_t worker_pid = -1;
    bool running = false;
};

#endif