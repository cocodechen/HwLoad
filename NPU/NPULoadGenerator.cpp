#ifdef HWLOAD_USE_NPU

#include "NPULoadGenerator.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <csignal>
#include <sys/wait.h>
#include <sys/types.h>

// 确保在 CMake 中定义了 NPU_SCRIPT_DIR
#ifndef NPU_SCRIPT_DIR
#define NPU_SCRIPT_DIR "./" 
#endif

NPULoadGenerator::NPULoadGenerator() : running(false), worker_pid(-1) {}

NPULoadGenerator::~NPULoadGenerator()
{
    stop();
}

std::string NPULoadGenerator::profileToString(ProfileType profile)
{
    switch(profile) {
        case ProfileType::Compute: return "compute";
        case ProfileType::Memory:  return "memory";
        case ProfileType::Data:    return "data";
        case ProfileType::IO:      throw std::runtime_error("NPU does not support IO load profile. Please use Compute, Memory, or Data.");
        case ProfileType::Random:   return "random";
        default: return "compute";
    }
}

std::string NPULoadGenerator::levelToString(LoadLevel level)
{
    switch(level) {
        case LoadLevel::Idle:      return "idle";
        case LoadLevel::Low:       return "low";
        case LoadLevel::Medium:    return "medium";
        case LoadLevel::High:      return "high";
        case LoadLevel::Saturated: return "saturated";
        default:                   return "idle";
    }
}

void NPULoadGenerator::start(ProfileType profile, LoadLevel level)
{
    if (running)return;
    running = true;

    // 2. 准备参数
    std::string s_profile = profileToString(profile);
    std::string s_level   = levelToString(level);
    std::string script_path = std::string(NPU_SCRIPT_DIR) + "/run_npu.sh";

    // 3. Fork 进程
    worker_pid = fork();
    if (worker_pid < 0)
    {
        running = false;
        throw std::runtime_error("[NPULoad] Failed to fork NPU worker process");
    }
    // --- Child Process ---
    else if (worker_pid == 0)
    {
        // [关键优化] 创建新的进程组
        // 这样 Shell 脚本和它启动的 Python 都会在这个组里
        // 父进程 kill(-pid) 时能把它们一锅端，防止僵尸进程
        setpgid(0, 0);

        // 准备 execvp 参数 (需要将 string vector 转为 char* array)
        std::vector<std::string> args = {
            script_path,        // argv[0]
            "--profile", s_profile,
            "--level",   s_level
        };
        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        for (auto& s : args)argv.push_back(s.data());  // C++17 起保证 '\0' 结尾，可写指针
        argv.push_back(nullptr);
        
        // 执行脚本
        execvp(argv[0], argv.data());

        // 只有出错才会走到这
        perror("[NPULoad] execvp failed");
        exit(1); 
    }
    else
    {
        // --- Parent Process ---
        std::cout << "[NPULoad] Worker started with PID: " << worker_pid << std::endl;
    }
}

void NPULoadGenerator::stop()
{
    if (!running) return;

    if (worker_pid > 0) {
        std::cout << "[NPULoad] Stopping worker PID: " << worker_pid << "..." << std::endl;
        
        // [关键优化] 发送信号给进程组 (-worker_pid)
        // 这会杀死 Shell 脚本以及它启动的 Python 进程
        kill(-worker_pid, SIGTERM);
        
        // 等待退出
        int status;
        waitpid(worker_pid, &status, 0);
        
        if (WIFEXITED(status)) {
            // 子进程正常退出 (exit code)
        } else if (WIFSIGNALED(status)) {
            // 子进程被信号杀死 (是我们预期的)
        }
        
        worker_pid = -1;
    }
    running = false;
    std::cout << "[NPULoad] Stopped." << std::endl;
}

#endif