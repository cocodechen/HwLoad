#include "LoadGenerator.hpp"

#ifdef HWLOAD_USE_CPU
    #include "CPU/CPULoadGenerator.hpp"
#endif

#ifdef HWLOAD_USE_GPU
    #include "GPU/GPULoadGenerator.hpp"
#endif

#ifdef HWLOAD_USE_NPU
    #include "NPU/NPULoadGenerator.hpp"
#endif

#include <iostream>
#include <string>
#include <memory>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>

std::atomic<bool> g_stop_requested{false};
static void signal_handler(int signal)
{
    if (signal == SIGINT)
    {
        g_stop_requested.store(true, std::memory_order_relaxed);
    }
}

// ---------- parse helpers ----------
ProfileType parseProfile(const std::string& s)
{
    if (s == "compute") return ProfileType::Compute;
    if (s == "memory")  return ProfileType::Memory;
    if (s == "io")      return ProfileType::IO;
    if (s == "data")    return ProfileType::Data;
    throw std::invalid_argument("unknown profile");
}

LoadLevel parseLevel(const std::string& s)
{
    if (s == "idle")       return LoadLevel::Idle;
    if (s == "low")        return LoadLevel::Low;
    if (s == "medium")     return LoadLevel::Medium;
    if (s == "high")       return LoadLevel::High;
    if (s == "saturated")  return LoadLevel::Saturated;
    throw std::invalid_argument("unknown level");
}

std::unique_ptr<LoadGenerator> parseDevice(const std::string& device)
{
    if (device == "cpu") {
#ifdef HWLOAD_USE_CPU
        return std::make_unique<CPULoadGenerator>();
#else
        throw std::runtime_error("CPU support not enabled at compile time");
#endif
    }

    if (device == "gpu") {
#ifdef HWLOAD_USE_GPU
        return std::make_unique<GPULoadGenerator>();
#else
        throw std::runtime_error("GPU support not enabled at compile time");
#endif
    }

    if (device == "npu") {
#ifdef HWLOAD_USE_NPU
        return std::make_unique<NPULoadGenerator>();
#else
        throw std::runtime_error("NPU support not enabled at compile time");
#endif
    }

    throw std::invalid_argument("Unknown device: " + device);
}


int main(int argc, char* argv[])
{
    // ---------- Parameter parsing ----------
    if (argc != 4) {
        std::cerr << "Usage:\n"
                  << "  loadgen <cpu|gpu|npu> <compute|memory|io|data> "
                  << "<idle|low|medium|high|saturated>\n";
        return 1;
    }

    std::string device  = argv[1];
    std::string profile = argv[2];
    std::string level   = argv[3];

    ProfileType profileType;
    LoadLevel   loadLevel;
    std::unique_ptr<LoadGenerator> generator;
    try
    {
        profileType = parseProfile(profile);
        loadLevel   = parseLevel(level);
        generator = parseDevice(device);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Argument error: " << e.what() << "\n";
        return 1;
    }

    // ---------- watch ----------
    std::cout << "Press Ctrl+C to stop...\n";
    std::signal(SIGINT, signal_handler);
    std::thread signal_watcher([&]{
		while (!g_stop_requested.load(std::memory_order_relaxed))
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		std::cout << "Stop generator\n";
		generator->stop();
	});

    // ---------- run ----------
    std::cout << "Starting load: "<< device << " / " << profile << " / " << level << "\n";
    try
    {
       generator->start(profileType, loadLevel);
    }
    catch(const std::exception& e)
    {
        std::cerr <<"Run Error: "<< e.what() << '\n';
        g_stop_requested.store(true, std::memory_order_relaxed);
    }
    signal_watcher.join();
	generator.reset();
    return 0;
}

