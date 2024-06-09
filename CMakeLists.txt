cmake_minimum_required(VERSION 3.16)
project(lns)

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Compiler flags for Release build
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -O3 -march=znver2")

# Include directories
include_directories(
    "inc"
    "inc/CBS"
    "inc/PIBT"
    "../LightGBM/include"
)

# Source files
file(GLOB SOURCES "src/*.cpp" "src/CBS/*.cpp" "src/PIBT/*.cpp")
add_executable(lns ${SOURCES})

# Static link libstdc++
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++")

# Find Boost
set(Boost_USE_STATIC_LIBS ON)  
find_package(Boost REQUIRED COMPONENTS program_options system filesystem)

# Find Eigen3 for PIBT
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Find OpenMP
# find_package(OpenMP REQUIRED)

# Linking libraries
target_link_libraries(lns 
    ${Boost_LIBRARIES} 
    Eigen3::Eigen
    "../LightGBM/lib_lightgbm.a"
    pthread
)
