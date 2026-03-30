cmake_minimum_required(VERSION 3.15)
project(GEMSSExamples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add GEMSS header-only library
add_subdirectory(GEMSS)

# Example executables
add_executable(multisphere_app main.cpp)
add_executable(multisphere_mesh main_mesh.cpp)
add_executable(multisphere_collision main_collision.cpp)
add_executable(multisphere_boost main_boost.cpp)
add_executable(multisphere_cube main_cube.cpp)
add_executable(multisphere_rectangle main_rectangle.cpp)

# Link GEMSS header-only library
foreach(target multisphere_app multisphere_mesh multisphere_collision multisphere_boost multisphere_cube multisphere_rectangle)
    target_link_libraries(${target} PRIVATE multisphere_lib)
    target_compile_options(${target} PRIVATE -O3 -march=native)
endforeach()
