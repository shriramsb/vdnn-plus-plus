cmake_minimum_required(VERSION 2.8)
project( NeuralNet )
include_directories( include /usr/include /usr/local/cuda/include /usr/include/x86_64-linux-gnu /usr/local/cuda/samples/common/inc)

link_directories( /usr/local/cuda/lib64/ )

find_package(CUDA)

cuda_add_executable( neural_net.out src/main.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu )
target_link_libraries( neural_net.out -lcudnn -lcublas -lcurand)

cuda_add_executable( mnist_test.out src/mnist_test.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu )
target_link_libraries( mnist_test.out -lcudnn -lcublas -lcurand)

cuda_add_executable( alexnet_test.out src/alexnet_test.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu )
target_link_libraries( alexnet_test.out -lcudnn -lcublas -lcurand)

cuda_add_executable( vgg_test.out src/vgg_test.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu )
target_link_libraries( vgg_test.out -lcudnn -lcublas -lcurand)

