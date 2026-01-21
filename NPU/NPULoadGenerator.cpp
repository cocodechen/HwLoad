#ifdef HWLOAD_USE_NPU

#include "NPULoadGenerator.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <csignal>
#include <sys/wait.h>
#include <sys/types.h>

NPULoadGenerator::NPULoadGenerator() : running(false), worker_pid(-1) {}

NPULoadGenerator::~NPULoadGenerator()
{
    stop();
}

void NPULoadGenerator::start(ProfileType profile, LoadLevel level)
{
    if (running) return;
    
    // 1. Idle 级别不启动任何进程
    if (level == LoadLevel::Idle)
    {
        return;
    }

    running = true;

    // 2. 参数映射：将枚举转换为字符串传给 Python
    std::string s_profile;
    switch(profile) {
        case ProfileType::Compute: s_profile = "compute"; break;
        case ProfileType::Memory:  s_profile = "memory"; break;
        case ProfileType::Data:    s_profile = "data"; break; 
        case ProfileType::IO:      s_profile = "memory"; break; // NPU无磁盘IO，映射为显存带宽负载
    }

    std::string s_level;
    switch(level) {
        case LoadLevel::Low:       s_level = "low"; break;
        case LoadLevel::Medium:    s_level = "medium"; break;
        case LoadLevel::High:      s_level = "high"; break;
        case LoadLevel::Saturated: s_level = "saturated"; break;
        default:                   s_level = "low"; break;
    }

    // 3. Fork 进程
    worker_pid = fork();

    if (worker_pid < 0) {
        std::cerr << "[NPULoad] Fork failed!" << std::endl;
        running = false;
        return;
    }

    if (worker_pid == 0) {
        // --- Child Process ---
        
        // 使用包装脚本，确保环境隔离 (假设脚本在当前目录)
        // 实际部署时建议使用绝对路径，例如 "/opt/my_app/run_npu.sh"
        std::string script = std::string(NPU_SCRIPT_DIR) + "/run_npu.sh";
        
        std::vector<const char*> args;
        args.push_back(script.c_str());
        args.push_back("--profile");
        args.push_back(s_profile.c_str());
        args.push_back("--level");
        args.push_back(s_level.c_str());
        args.push_back(nullptr); // 必须以 NULL 结尾

        // 替换进程映像
        execvp(script.c_str(), const_cast<char* const*>(args.data()));

        // 如果代码走到这，说明 execvp 失败了
        perror("[NPULoad] Failed to execvp run_npu.sh");
        exit(1);
    } 
    
    // --- Parent Process ---
    // 父进程只需记录 PID，非阻塞返回
}

void NPULoadGenerator::stop()
{
    if (!running) return;

    if (worker_pid > 0) {
        // 发送 SIGTERM 信号
        kill(worker_pid, SIGTERM);
        
        // 等待子进程退出，回收僵尸进程
        int status;
        waitpid(worker_pid, &status, 0);
        
        worker_pid = -1;
    }
    running = false;
}

#endif