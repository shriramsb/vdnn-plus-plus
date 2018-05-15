#include <iostream>
#include <vector>
#include <string>

#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <helper_cuda.h>
#include "user_iface.h"
#include "layer_params.h"
#include "utils.h"

#ifndef NEURAL_NET
#define NEURAL_NET
class NeuralNet {
public:
	void **layer_input, **dlayer_input, **params;
	int *y, *pred_y;
	float *loss;
	float softmax_eps;
	void *one_vec;
	float init_std_dev;

	std::vector<LayerOp> layer_type;
	int num_layers;
	cudnnHandle_t cudnn_handle;
	cublasHandle_t cublas_handle;
	curandGenerator_t curand_gen;

	cudnnDataType_t data_type;
	size_t data_type_size;
	cudnnTensorFormat_t tensor_format;
	int batch_size;

	size_t free_bytes, total_bytes;
	size_t workspace_size;
	void *workspace;

	int input_channels, input_h, input_w;
	int num_classes;

	float *h_loss;
	int *h_pred_y;
	ConvAlgo conv_algo;


	NeuralNet(std::vector<LayerSpecifier> &layers, DataType data_type, int batch_size, TensorFormat tensor_format, 
				long long dropout_seed, float softmax_eps, float init_std_dev, ConvAlgo conv_algo = CONV_ALGO_AUTO);

	void getLoss(void *X, int *y, bool train = true, int *correct_count = NULL, float *loss = NULL);

	void compareOutputCorrect(int *correct_count, int *y);

	float computeLoss();

};

#endif