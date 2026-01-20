#ifdef HWLOAD_USE_CPU

#include "CPULoadGenerator.hpp"

#include <chrono>
#include <cmath>
#include <vector>
#include <stdexcept>

#ifdef __linux__
    #include <unistd.h>
    #include <fcntl.h>
#endif

CPULoadGenerator::CPULoadGenerator(): running(false){}

void CPULoadGenerator::start(ProfileType profile,LoadLevel level)
{
    cur_level=level;
    running = true;
    if (profile == ProfileType::Compute) {
        worker = std::thread(&CPULoadGenerator::runCompute, this);
    }
    else if (profile == ProfileType::Memory) {
        worker = std::thread(&CPULoadGenerator::runMemory, this);
    }
    else if (profile == ProfileType::Data) {
        worker = std::thread(&CPULoadGenerator::runData, this);
    }
    else if (profile == ProfileType::IO)
    {
#ifdef __linux__
        worker = std::thread(&CPULoadGenerator::runIO, this);
#else
        throw std::runtime_error("CPU IO load only supported on Linux");
#endif
    }
}

void CPULoadGenerator::stop()
{
    running = false;
    if (worker.joinable())worker.join();
}

/*busy loop + sleep*/
void CPULoadGenerator::runCompute()
{
    using namespace std::chrono;

    auto busy_idle = [&](milliseconds busy, milliseconds idle) {
        while (running) {
            auto start = steady_clock::now();
            while (steady_clock::now() - start < busy) {
                volatile double x = 0;
                for (int i = 0; i < 10000; ++i)
                    x += std::sin(i);
            }
            if (idle.count() > 0)
                std::this_thread::sleep_for(idle);
        }
    };

    switch (cur_level)
    {
        case LoadLevel::Idle:
            busy_idle(0ms, 100ms);
            break;
        case LoadLevel::Low:
            busy_idle(10ms, 90ms);
            break;
        case LoadLevel::Medium:
            busy_idle(40ms, 60ms);
            break;
        case LoadLevel::High:
            busy_idle(80ms, 20ms);
            break;
        case LoadLevel::Saturated:
            while (running)
            {
                volatile double x = 0;
                for (int i = 0; i < 100000; ++i)
                    x += std::sqrt(i);
            }
            break;
    }
}

void CPULoadGenerator::runMemory()
{
    constexpr size_t BUF_SIZE = 256 * 1024 * 1024; // 256MB
    std::vector<char> buffer(BUF_SIZE);

    auto touch = [&](size_t stride) {
        for (size_t i = 0; i < BUF_SIZE; i += stride) {
            buffer[i]++;
        }
    };

    size_t stride = 64; // cache line
    while (running)
    {
        switch (cur_level)
        {
        case LoadLevel::Idle:
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            break;
        case LoadLevel::Low:
            touch(512);
            break;
        case LoadLevel::Medium:
            touch(256);
            break;
        case LoadLevel::High:
            touch(128);
            break;
        case LoadLevel::Saturated:
            touch(64);
            break;
        }
    }
}

void CPULoadGenerator::runData()
{
    constexpr size_t N = 64 * 1024 * 1024; // 64M ints
    std::vector<int> data(N, 1);

    auto process = [&](size_t step)
    {
        for (size_t i = 0; i < N; i += step) {
            data[i] = (data[i] * 31 + 7) % 1000003;
        }
    };

    while (running)
    {
        switch (cur_level)
        {
        case LoadLevel::Idle:
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            break;
        case LoadLevel::Low:
            process(1024);
            break;
        case LoadLevel::Medium:
            process(256);
            break;
        case LoadLevel::High:
            process(64);
            break;
        case LoadLevel::Saturated:
            process(1);
            break;
        }
    }
}


#ifdef __linux__
void CPULoadGenerator::runIO()
{
    constexpr size_t BUF_SIZE = 4 * 1024 * 1024; // 4MB
    std::vector<char> buffer(BUF_SIZE, 1);

    int fd = open("cpu_io_tmp.bin", O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd < 0) return;

    while (running) {
        switch (cur_level) {
        case LoadLevel::Idle:
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            break;
        case LoadLevel::Low:
            write(fd, buffer.data(), 512 * 1024);
            fsync(fd);
            break;
        case LoadLevel::Medium:
            write(fd, buffer.data(), 1 * 1024 * 1024);
            fsync(fd);
            break;
        case LoadLevel::High:
            write(fd, buffer.data(), 2 * 1024 * 1024);
            fsync(fd);
            break;
        case LoadLevel::Saturated:
            write(fd, buffer.data(), BUF_SIZE);
            fsync(fd);
            break;
        }
    }

    close(fd);
    unlink("cpu_io_tmp.bin");
}
#endif

#endif

