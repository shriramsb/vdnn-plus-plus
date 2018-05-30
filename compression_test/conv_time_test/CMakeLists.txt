cmake_minimum_required(VERSION 2.8)
project( ConvTimeTest )
include_directories( include /usr/include /usr/local/cuda/include /usr/include/x86_64-linux-gnu /usr/local/cuda/samples/common/inc)

find_package(CUDA)

cuda_add_executable( conv_time conv_time.cu )
target_link_libraries( conv_time ${OpenCV_LIBS} -lcudnn "/usr/local/cuda/lib64/libcublas.so" "/usr/local/cuda/lib64/libcurand.so")