cmake_minimum_required(VERSION 2.8)
project( vDNNNeuralNet )
include_directories( include )

link_directories( /usr/local/cuda/lib64/ )

find_package(CUDA)

cuda_add_executable( mem_alloc.out src/mem_alloc.cu src/cnmem.cpp )
target_link_libraries( mem_alloc.out -lcudnn )