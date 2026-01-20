#pragma once

#include <memory>
#include "LoadGenerator.hpp"

class LoadController
{
public:
    explicit LoadController(std::unique_ptr<LoadGenerator> generator);

    // 固定强度运行（当前阶段用这个）
    void run(ProfileType profile, LoadLevel level);

    // 停止
    void stop();

private:
    std::unique_ptr<LoadGenerator> generator;
};