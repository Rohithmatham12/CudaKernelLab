cmake_minimum_required(VERSION 3.24)
project(CudaKernelLab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_library(cuda_kernels
  src/kernels.cu
  src/host.cpp
)
target_include_directories(cuda_kernels PUBLIC include)
target_compile_options(cuda_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
)

add_executable(kernel_tests tests/kernel_tests.cpp)
target_link_libraries(kernel_tests PRIVATE cuda_kernels)

add_executable(benchmark_kernels benchmarks/benchmark_kernels.cpp)
target_link_libraries(benchmark_kernels PRIVATE cuda_kernels)

enable_testing()
add_test(NAME kernel_tests COMMAND kernel_tests)
set_tests_properties(kernel_tests PROPERTIES SKIP_RETURN_CODE 77)
