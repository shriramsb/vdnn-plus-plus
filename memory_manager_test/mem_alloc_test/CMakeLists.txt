cmake_minimum_required(VERSION 2.8)
project( vDNNNeuralNet )
include_directories( include cnmem/include )

link_directories( /usr/local/cuda/lib64/ cnmem/build )

find_package(CUDA)

cuda_add_executable( mem_alloc.out mem_alloc.cu)
target_link_libraries( mem_alloc.out -lcudnn -lcnmem )