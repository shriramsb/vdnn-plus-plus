cmake_minimum_required(VERSION 2.8)
project( vDNNNeuralNet )
include_directories( include /usr/include /usr/local/cuda/include /usr/include/x86_64-linux-gnu /usr/local/cuda/samples/common/inc)

link_directories( /usr/local/cuda/lib64/ )

find_package(CUDA)

OPTION(GTX970 "Set if trained on GTX970" OFF)
OPTION(DEBUG_FPROP_ALLOC "Set if need to dump cnmem state at the end of forward prop" OFF)
OPTION(DEBUG_FPROP_ALLOC_ITERS "Set if need to dump cnmem state at the start of every iteration of forward prop" OFF)
OPTION(ALLOC_NON_OFFLOAD_RIGHT "Set if need to allocate non offloaded layers from right in fprop" OFF)


IF(GTX970)
	ADD_DEFINITIONS(-DGTX970)
ENDIF(GTX970)

IF(DEBUG_FPROP_ALLOC)
	ADD_DEFINITIONS(-DDEBUG_FPROP_ALLOC)
ENDIF(DEBUG_FPROP_ALLOC)
IF(DEBUG_FPROP_ALLOC_ITERS)
	ADD_DEFINITIONS(-DDEBUG_FPROP_ALLOC_ITERS)
ENDIF(DEBUG_FPROP_ALLOC_ITERS)
IF(ALLOC_NON_OFFLOAD_RIGHT)
	ADD_DEFINITIONS(-DALLOC_NON_OFFLOAD_RIGHT)
ENDIF(ALLOC_NON_OFFLOAD_RIGHT)

cuda_add_executable( neural_net_vdnn.out src/main.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu src/cnmem.cpp)
target_link_libraries( neural_net_vdnn.out -lcudnn -lcublas -lcurand -lpthread)

cuda_add_executable( mnist_test.out src/mnist_test.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu src/cnmem.cpp)
target_link_libraries( mnist_test.out -lcudnn -lcublas -lcurand -lpthread)

cuda_add_executable( alexnet_test.out src/alexnet_test.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu src/cnmem.cpp)
target_link_libraries( alexnet_test.out -lcudnn -lcublas -lcurand -lpthread)

cuda_add_executable( vgg_test.out src/vgg_test.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu src/cnmem.cpp)
target_link_libraries( vgg_test.out -lcudnn -lcublas -lcurand -lpthread)

cuda_add_executable( slow_conv.out src/slow_conv.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu src/cnmem.cpp)
target_link_libraries( slow_conv.out -lcudnn -lcublas -lcurand -lpthread)


