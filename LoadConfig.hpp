#pragma once

#include <cstdint>
#include <string>
#include <stdexcept>

enum class ProfileType
{
    Compute,   // 计算密集
    Memory,    // 内存/带宽密集
    Data,      // 混合
    IO,        // IO
    Random,     // 随机模拟真实负载
    Real   // 实时负载
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

inline ProfileType str2Profile(const std::string& s)
{
    if (s == "compute") return ProfileType::Compute;
    if (s == "memory")  return ProfileType::Memory;
    if (s == "io")      return ProfileType::IO;
    if (s == "data")    return ProfileType::Data;
    if (s == "random")  return ProfileType::Random; 
    if (s == "real")    return ProfileType::Real;
    throw std::invalid_argument("unknown profile: " + s);
}

inline LoadLevel str2Level(const std::string& s)
{
    if (s == "idle")       return LoadLevel::Idle;
    if (s == "low")        return LoadLevel::Low;
    if (s == "medium")     return LoadLevel::Medium;
    if (s == "high")       return LoadLevel::High;
    if (s == "saturated")  return LoadLevel::Saturated;
    throw std::invalid_argument("unknown level: " + s);
}

inline const char* profile2Str(ProfileType p)
{
    switch (p) {
    case ProfileType::Compute: return "Compute";
    case ProfileType::Memory:  return "Memory";
    case ProfileType::Data:    return "Data";
    case ProfileType::IO:      return "IO";
    case ProfileType::Random:  return "Random";
    case ProfileType::Real:    return "Real";
    default:                   return "Unknown";
    }
}

inline const char* level2Str(LoadLevel l)
{
    switch (l) {
    case LoadLevel::Idle:       return "Idle";
    case LoadLevel::Low:        return "Low";
    case LoadLevel::Medium:     return "Medium";
    case LoadLevel::High:       return "High";
    case LoadLevel::Saturated:  return "Saturated";
    default:                    return "Unknown";
    }
}
