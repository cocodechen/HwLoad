#pragma once

#include "LoadConfig.hpp"
#include <chrono>
#include <cmath>
#include <random>

class LoadGenerator
{
public:
    LoadGenerator():cur_level(LoadLevel::Idle){}
    
    virtual ~LoadGenerator() = default;

    // 启动负载
    virtual void start(ProfileType profile,LoadLevel level) = 0;

    // 停止负载
    virtual void stop() = 0;

protected:
    LoadLevel cur_level;
};

// --- 辅助函数：获取模拟波形负载强度 (0.0 - 1.0) ---
// 对应 Python 中的 get_wave_intensity
inline double get_wave_intensity(double offset_seed)
{
    auto now = std::chrono::system_clock::now();
    double t = std::chrono::duration<double>(now.time_since_epoch()).count();

    double slow = (std::sin(t / 60.0 + offset_seed) + 1.0) / 2.0;
    double fast = (std::sin(t / 5.0 + offset_seed * 3.0) + 1.0) / 2.0;
    
    // 简单的随机噪声
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double noise = dis(gen);

    double val = 0.5 * slow + 0.3 * fast + 0.2 * noise;
    return std::max(0.0, std::min(val, 1.0));
}

