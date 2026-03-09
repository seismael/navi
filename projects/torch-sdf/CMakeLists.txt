cmake_minimum_required(VERSION 3.18)
project(torch_sdf LANGUAGES CXX CUDA)

# Find PyTorch
find_package(Torch REQUIRED)

# Define the Python extension
set(PYBIND11_FINDPACKAGE ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(torch_sdf_backend src/bindings.cpp src/kernel.cu)

# Link LibTorch and CUDA
target_link_libraries(torch_sdf_backend PRIVATE "${TORCH_LIBRARIES}")
target_compile_features(torch_sdf_backend PRIVATE cxx_std_17)

# CUDA Architecture Configuration
set_target_properties(torch_sdf_backend PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
)

# Suppress common warnings
if (MSVC)
    target_compile_options(torch_sdf_backend PRIVATE /wd4244 /wd4267 /wd4819)
else()
    target_compile_options(torch_sdf_backend PRIVATE -O3 -ffast-math)
endif()
