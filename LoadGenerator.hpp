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
inline double get_wave_intensity(double offset_seed) {
    auto now = std::chrono::system_clock::now();
    // 获取秒数
    double t = std::chrono::duration<double>(now.time_since_epoch()).count();

    // 1. 基础趋势波 (周期很长，约 40秒)
    // 范围 [-0.5, 0.5]
    double slow_wave = 0.5 * std::sin(t / 20.0 + offset_seed);

    // 2. 活跃波动波 (周期较短，约 5秒)
    // 范围 [-0.2, 0.2]
    double fast_wave = 0.2 * std::sin(t / 2.5 + offset_seed * 2.0);

    // 3. 随机抖动 (让曲线看起来不那么平滑)
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> noise_dist(-0.05, 0.05);
    double noise = noise_dist(gen);

    // 叠加: 基准值 0.5 + 慢波 + 快波 + 噪声
    // 结果大概率落在 [0.1, 0.9] 之间，偶尔触顶或触底
    double val = 0.5 + slow_wave + fast_wave + noise;

    // 钳制在 [0.0, 1.0] 安全范围
    return std::max(0.01, std::min(val, 1.0));
}

