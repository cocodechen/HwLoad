#pragma once

#ifdef HWLOAD_USE_NPU

#include "LoadGenerator.hpp"
#include <unistd.h>
#include <string>
#include <vector>

class NPULoadGenerator : public LoadGenerator
{
public:
    NPULoadGenerator();
    ~NPULoadGenerator();

    void start(ProfileType profile, LoadLevel level) override;
    void stop() override;

private:
    // 进程 ID
    pid_t worker_pid = -1;
    bool running = false;

    // 将枚举转换为 Python 脚本接受的字符串参数
    std::string profileToString(ProfileType profile);
    std::string levelToString(LoadLevel level);
};

#endif