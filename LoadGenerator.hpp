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
// --- 0.5单峰 ---
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

// 左右双峰
// inline double get_wave_intensity(double offset_seed) {
//     auto now = std::chrono::system_clock::now();
//     double t = std::chrono::duration<double>(now.time_since_epoch()).count();

//     // =========================================================
//     // 1. 宏观趋势 (Macro Trend) - 决定“忙闲阶段”
//     // =========================================================
//     // 周期：约 60秒 (t / 10.0)
//     // 作用：让基准线在 [0.2, 0.8] 之间缓慢游走。
//     // 当它为正时，整体处于“高负载区”；为负时，处于“低负载区”。
//     double macro_trend = 0.35 * std::sin(t / 10.0 + offset_seed * 0.5);
    
//     // 动态基准线：不再固定是 0.5，而是随时间变化
//     double dynamic_base = 0.5 + macro_trend; 

//     // =========================================================
//     // 2. 活跃波动 (Micro Fluctuation) - 模拟“正在处理任务”
//     // =========================================================
//     // 周期：约 3秒，频率较快
//     // 幅度：0.15，让曲线在基准线上下跳动，但不至于由于幅度太大而一直被截断
//     double active_wave = 0.15 * std::sin(t / 0.5 + offset_seed * 2.0);

//     // =========================================================
//     // 3. 随机突发/噪声 (Noise & Spikes)
//     // =========================================================
//     static thread_local std::mt19937 gen(std::random_device{}());
//     std::uniform_real_distribution<> noise_dist(-0.05, 0.05);
    
//     // 偶尔加入一个“突发任务” (Spike)
//     // 利用 fmod 模拟每 13 秒左右可能出现一次短暂的脉冲
//     double spike = 0.0;
//     // 使用 t 的小数部分或其他周期函数来触发
//     if (std::sin(t / 2.0 + offset_seed) > 0.95) { 
//         spike = 0.3; // 突然增加 30% 负载
//     }

//     // =========================================================
//     // 4. 合成
//     // =========================================================
//     // 结果 = 动态基准 + 活跃波动 + 噪声 + 突发
//     double val = dynamic_base + active_wave + noise_dist(gen) + spike;

//     // 钳制
//     return std::max(0.01, std::min(val, 1.0));
// }
