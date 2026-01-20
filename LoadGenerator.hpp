#pragma once

#include "LoadConfig.hpp"

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