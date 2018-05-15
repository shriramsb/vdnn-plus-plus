cmake_minimum_required(VERSION 2.8)
project( vDNNNeuralNet )
include_directories( include /usr/include /usr/local/cuda/include /usr/include/x86_64-linux-gnu /usr/local/cuda/samples/common/inc cnmem/include )

link_directories( /usr/local/cuda/lib64/ cnmem/build )

find_package(CUDA)

cuda_add_executable( neural_net_vdnn src/main.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu )
target_link_libraries( neural_net_vdnn -lcudnn -lcublas -lcurand -lcnmem -lpthread)

cuda_add_executable( mnist_test src/mnist_test.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu )
target_link_libraries( mnist_test -lcudnn -lcublas -lcurand -lcnmem -lpthread)

cuda_add_executable( alexnet_test src/alexnet_test.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu )
target_link_libraries( alexnet_test -lcudnn -lcublas -lcurand -lcnmem -lpthread)

cuda_add_executable( vgg_test src/vgg_test.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu )
target_link_libraries( vgg_test -lcudnn -lcublas -lcurand -lcnmem -lpthread)

cuda_add_executable( slow_conv src/slow_conv.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu )
target_link_libraries( slow_conv -lcudnn -lcublas -lcurand -lcnmem -lpthread)


