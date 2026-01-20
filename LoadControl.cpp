#include "LoadControl.hpp"

LoadController::LoadController(std::unique_ptr<LoadGenerator> gen)
    : generator(std::move(gen)) {}

void LoadController::run(ProfileType profile, LoadLevel level)
{
    generator->start(profile,level);
}

void LoadController::stop()
{
    generator->stop();
}