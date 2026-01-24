#pragma once

#include <cstdint>
#include <string>

enum class ProfileType
{
    Compute,   // 计算密集
    Memory,    // 内存/带宽密集
    Data,      // 混合
    IO,        // IO
    Random     // [NEW] 随机模拟真实负载
};

enum class LoadLevel
{
    Idle,       
    Low,        
    Medium,     
    High,       
    Saturated   
};

struct LoadTask
{
    std::string device;
    ProfileType profile;
    LoadLevel level;
};