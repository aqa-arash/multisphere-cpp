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
    target_link_libraries(${target} PRIVATE GEMSS)
    target_compile_options(${target} PRIVATE -O3 -march=native)
endforeach()



# If you are compiling it for multiple interconnected projects using fetch, consider adding the following lines : 

#include(FetchContent)

#FetchContent_Declare(
#    GEMSS
#    GIT_REPOSITORY https://github.com/aqa-arash/GEMSS.git
#    GIT_TAG        main
#    # The CMakeLists.txt file is located in the folder src
#    SOURCE_SUBDIR  src/GEMSS

    # Inject 'inline' using the correct relative paths from the repo root
#    PATCH_COMMAND  sed -i "s/void ThreadPool::start(/inline void ThreadPool::start(/g" src/GEMSS/thirdparty/edt.hpp &&
#                   sed -i "s/void squared_edt_1d_parabolic(/inline void squared_edt_1d_parabolic(/g" src/GEMSS/thirdparty/edt.hpp &&
#                   sed -i "s/void _squared_edt_1d_parabolic(/inline void _squared_edt_1d_parabolic(/g" src/GEMSS/thirdparty/edt.hpp &&

#)

#FetchContent_MakeAvailable(GEMSS)
#
#
#