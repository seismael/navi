cmake_minimum_required(VERSION 3.18)
project(VoxelDagCompiler VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC-safe strict compilation flags
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# --- Assimp via FetchContent (no manual install required) ---
include(FetchContent)
FetchContent_Declare(
    assimp
    GIT_REPOSITORY https://github.com/assimp/assimp.git
    GIT_TAG        v5.4.3
    GIT_SHALLOW    TRUE
)
# Disable Assimp extras we don't need (faster build)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ZLIB ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(assimp)

# Includes
include_directories(src third_party)

# Executable
add_executable(voxel-dag
    src/main.cpp
    src/compiler.cpp
)

target_link_libraries(voxel-dag PRIVATE assimp::assimp)
