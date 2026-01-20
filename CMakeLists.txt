cmake_minimum_required(VERSION 3.20)

project(LoadGenerator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== Options =====
option(HWLOAD_USE_CPU "Enable CPU load generator" ON)
option(HWLOAD_USE_GPU "Enable GPU load generator" ON)
option(HWLOAD_USE_NPU "Enable NPU load generator" OFF)

# ===== Sources =====
set(SOURCES main.cpp)

# ===== Target =====
add_executable(loadgen ${SOURCES})

target_include_directories(loadgen
    PUBLIC
        ${PROJECT_SOURCE_DIR}
)

# ===== CPU =====
if (HWLOAD_USE_CPU)
    message(STATUS "CPU load generator enabled")
    target_sources(loadgen PRIVATE CPU/CPULoadGenerator.cpp)
    target_compile_definitions(loadgen PRIVATE HWLOAD_USE_CPU)
endif()

# ===== GPU =====
if (HWLOAD_USE_GPU)
    message(STATUS "GPU load generator enabled")

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED) #会到宏定义的变量、默认路径、环境变量去寻找

    list(APPEND SOURCES GPU/GPULoadGenerator.cu)
    target_sources(loadgen PRIVATE GPU/GPULoadGenerator.cu)

    target_compile_definitions(loadgen PRIVATE HWLOAD_USE_GPU)

    set_target_properties(loadgen PROPERTIES
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES native   
    )

    target_link_libraries(loadgen PRIVATE CUDA::cudart)
endif()

# ===== NPU =====
if (HWLOAD_USE_NPU)
    message(STATUS "NPU load generator enabled")
endif()

# ===== Platform =====
if (WIN32)
    target_compile_definitions(loadgen PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
