#pragma once

#include <cstdint>

enum class ProfileType
{
    Compute,   // 计算密集：CPU、GPU、NPU
    Memory,    // 内存/带宽密集：CPU、GPU
    Data,      // 泛指大规模数据处理=Compute+Memory：CPU、GPU、NPU
    IO         // 磁盘/网络 ：CPU
};

enum class LoadLevel
{
    Idle,       // 空闲 / 几乎无负载
    Low,        // 低负载（轻微扰动）
    Medium,     // 中等负载（可观但不饱和）
    High,       // 高负载（接近瓶颈）
    Saturated   // 满载 / 压测极限
};
