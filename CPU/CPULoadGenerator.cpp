#ifdef HWLOAD_USE_CPU

#include "CPULoadGenerator.hpp"

#include <chrono>
#include <cmath>
#include <vector>
#include <stdexcept>

#ifdef __linux__
    #include <unistd.h>
    #include <fcntl.h>
    #include <sys/mman.h>   
    #include <sys/stat.h>
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

void CPULoadGenerator::runCompute()
{
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());

    static std::atomic<uint64_t> sink{0};

    auto worker = [&](int intensity) {
        double x = 1.0;
        while (running) {
            for (int i = 0; i < intensity; ++i) {
                x = x * 1.0000001
                + std::sin(x)
                - std::log(x + 1.0)
                + std::sqrt(x + 0.5);

                if ((i & 15) == 0)
                    x *= 0.999;
            }
            // 防止编译器消除整个循环
            sink.fetch_add(static_cast<uint64_t>(x),std::memory_order_relaxed);
        }
    };

    int intensity = 0;
    switch (cur_level) {
    case LoadLevel::Idle:      intensity = 0; break;
    case LoadLevel::Low:       intensity = 2'000; break;
    case LoadLevel::Medium:    intensity = 10'000; break;
    case LoadLevel::High:      intensity = 50'000; break;
    case LoadLevel::Saturated: intensity = 200'000; break;
    }

    if (intensity == 0) {
        while (running)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return;
    }

    std::vector<std::thread> threads;
    for (unsigned i = 0; i < cores; ++i)
        threads.emplace_back(worker, intensity);

    for (auto& t : threads)
        t.join();
}

void CPULoadGenerator::runMemory()
{
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());

    const size_t total_size =
        (cur_level == LoadLevel::Low)       ? 32  * 1024 * 1024 :
        (cur_level == LoadLevel::Medium)    ? 128 * 1024 * 1024 :
        (cur_level == LoadLevel::High)      ? 512 * 1024 * 1024 :
        (cur_level == LoadLevel::Saturated) ? 1024 * 1024 * 1024 :
                                              0;

    if (total_size == 0) {
        while (running)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return;
    }

    std::vector<char> buffer(total_size);
    const size_t chunk = total_size / threads;

    const size_t seq_stride = 64;
    const size_t rnd_stride = 4096; // page 粒度扰动

    auto worker = [&](size_t begin, size_t end) {
        while (running) {
            // 顺序访问（主体）
            for (size_t i = begin; i < end; i += seq_stride)
                buffer[i]++;

            // 固定比例随机扰动（不 shuffle）
            for (size_t i = begin; i < end; i += rnd_stride)
                buffer[i ^ 0x5a5a]++;
        }
    };

    std::vector<std::thread> ts;
    for (unsigned i = 0; i < threads; ++i) {
        size_t b = i * chunk;
        size_t e = (i == threads - 1) ? total_size : b + chunk;
        ts.emplace_back(worker, b, e);
    }

    for (auto& t : ts)
        t.join();
}

void CPULoadGenerator::runData()
{
    const size_t N =
        (cur_level == LoadLevel::Low)       ? 1  * 1024 * 1024 :
        (cur_level == LoadLevel::Medium)    ? 8  * 1024 * 1024 :
        (cur_level == LoadLevel::High)      ? 32 * 1024 * 1024 :
        (cur_level == LoadLevel::Saturated) ? 128 * 1024 * 1024 :
                                              0;

    if (N == 0) {
        while (running)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return;
    }

    struct Node {
        int value;
        int next;
        char pad[64 - 8]; // cache line sized
    };

    std::vector<Node> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i].value = static_cast<int>(i);
        data[i].next  = (i * 1315423911u) % N; // pseudo-random chain
    }

    int idx = 0;
    volatile int sink = 0;

    using clock = std::chrono::steady_clock;

    while (running) {
        auto start = clock::now();

        while (clock::now() - start < std::chrono::milliseconds(20)) {
            Node& n = data[idx];

            // -------- memory part --------
            int v = n.value;

            // -------- compute part --------
            // 小 compute kernel，防止被优化掉
            for (int k = 0; k < 8; ++k) {
                v = v * 31 + (v >> 3) + k;
                v ^= (v << 7);
            }

            // -------- write-back --------
            n.value = v;
            sink += v;

            idx = n.next;
        }

        // 明确的 idle window，避免 scheduler 噪声
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

#ifdef __linux__
void CPULoadGenerator::runIO()
{
    const size_t map_size =
        (cur_level == LoadLevel::Low)       ? 16  * 1024 * 1024 :
        (cur_level == LoadLevel::Medium)    ? 64  * 1024 * 1024 :
        (cur_level == LoadLevel::High)      ? 256 * 1024 * 1024 :
        (cur_level == LoadLevel::Saturated) ? 1024 * 1024 * 1024 :
                                              0;

    if (map_size == 0) {
        while (running)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return;
    }

    int fd = open("/tmp/cpu_io_tmp.bin", O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd < 0) return;

    ftruncate(fd, map_size);

    char* p = static_cast<char*>(
        mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));

    const size_t stride = 4096;

    while (running) {
        for (size_t i = 0; i < map_size; i += stride * 16)
            p[i]++;

        msync(p, map_size, MS_ASYNC);

        // IO 场景必须给 CPU 明确 idle
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    munmap(p, map_size);
    close(fd);
    unlink("/tmp/cpu_io_tmp.bin");
}
#endif

#endif

