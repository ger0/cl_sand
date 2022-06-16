cmake_minimum_required(VERSION 3.7)

project(gasand-cl)

find_package(SDL2 REQUIRED)
find_library(CL_LIB OpenCL)
include_directories(${SDL2_INCLUDE_DIRS})

add_executable(gasand-cl main.cpp)
target_link_libraries(gasand-cl ${SDL2_LIBRARIES})
target_link_libraries(gasand-cl ${CL_LIB})
