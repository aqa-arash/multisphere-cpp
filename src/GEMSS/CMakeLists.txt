cmake_minimum_required(VERSION 3.15)
project(GEMSS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Build Options ---
option(GEMSS_USE_OPENMP "Use OpenMP for parallelization" ON)
option(GEMSS_DEBUG "Enable console debug prints (GEMSS_DEBUG macro)" OFF)

# --- Default Build Type to Release ---
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, defaulting to Release for maximum performance.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# --- Dependencies ---
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if(GEMSS_USE_OPENMP)
    find_package(OpenMP)
    if(NOT OpenMP_CXX_FOUND)
        message(WARNING "OpenMP requested but not found. Compiling single-threaded.")
        set(GEMSS_USE_OPENMP OFF)
    endif()
endif()

# --- Library Target ---
add_library(GEMSS INTERFACE)
target_include_directories(GEMSS INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/igl>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(GEMSS INTERFACE Eigen3::Eigen)

if(GEMSS_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(GEMSS INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(GEMSS INTERFACE HAVE_OPENMP)
endif()

if(GEMSS_DEBUG)
    target_compile_definitions(GEMSS INTERFACE GEMSS_DEBUG)
endif()

# Only header-only library, no executables or targets here

# --- Installation Rules ---
include(GNUInstallDirs)
install(TARGETS GEMSS
    EXPORT GEMSSTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY . DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
