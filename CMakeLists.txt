#cmake_minimum_required(VERSION 3.17)
#project(gpu_graph_algorithms CUDA)
#
#set(CMAKE_CUDA_STANDARD 14)
#
#add_executable(gpu_graph_algorithms main.cu)
#
#set_target_properties(
#        gpu_graph_algorithms
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
#
#
cmake_minimum_required(VERSION 3.14)
project(hw3_cuda LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
add_executable(seq_bellman bellman.cpp)
add_executable(par_bellman par_bellman.cu)