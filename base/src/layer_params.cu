#include "layer_params.h"

void ConvLayerParams::initializeValues(cudnnHandle_t cudnn_handle, ConvDescriptor *user_params, cudnnDataType_t data_type, 
									int batch_size, cudnnTensorFormat_t tensor_format, size_t data_type_size, LayerDimension &output_size) {
	// create tensor, filter, conv descriptor
	checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&output_tensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&bias_desc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

	C_in = user_params->input_channels;
	C_out = user_params->output_channels;
	filter_h = user_params->kernel_h;
	filter_w = user_params->kernel_w;

	checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, 
										batch_size, user_params->input_channels, user_params->input_h, user_params->input_w));


	checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc, data_type, tensor_format, 
										user_params->output_channels, user_params->input_channels, user_params->kernel_h, user_params->kernel_w));

	int dilation_h = 1, dilation_w = 1;
	checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc, user_params->pad_h, user_params->pad_w, 
												user_params->stride_y, user_params->stride_x,
												dilation_h, dilation_w, 
												CUDNN_CROSS_CORRELATION, data_type));

	int output_batch_size, output_channels, output_h, output_w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_tensor, filter_desc,
													&output_batch_size, &output_channels, &output_h, &output_w));

	checkCUDNN(cudnnSetTensor4dDescriptor(output_tensor, tensor_format, data_type, 
										output_batch_size, output_channels, output_h, output_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(bias_desc, tensor_format, data_type, 
										1, output_channels, 1, 1));

	fwd_req_count = 10;
	fwd_perf = (cudnnConvolutionFwdAlgoPerf_t *)malloc(fwd_req_count * sizeof(cudnnConvolutionFwdAlgoPerf_t));
	checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn_handle, 
													input_tensor, filter_desc, conv_desc, output_tensor, 
													fwd_req_count, &fwd_ret_count, fwd_perf));

	bwd_filter_req_count = 10;
	bwd_filter_perf = (cudnnConvolutionBwdFilterAlgoPerf_t *)malloc(bwd_filter_req_count * sizeof(cudnnConvolutionBwdFilterAlgoPerf_t));
	checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(cudnn_handle, 
															input_tensor, output_tensor, conv_desc, filter_desc, 
															bwd_filter_req_count, &bwd_filter_ret_count, bwd_filter_perf));

	// std::cout << "Printing bwdfilter conv algo perf\n";
	// for (int i = 0; i < bwd_filter_ret_count; i++) {
	// 	std::cout << i << std::endl;
	// 	std::cout << "algo: " << bwd_filter_perf[i].algo << std::endl;
	// 	std::cout << "status: " << cudnnGetErrorString(bwd_filter_perf[i].status) << std::endl;
	// 	std::cout << "time(ms): " << bwd_filter_perf[i].time << std::endl;
	// 	std::cout << "memory(bytes): " << bwd_filter_perf[i].memory << std::endl;
	// 	std::cout << "mathType: " << bwd_filter_perf[i].mathType << std::endl;
	// 	std::cout << std::endl;
	// }
	bwd_data_req_count = 10;
	bwd_data_perf = (cudnnConvolutionBwdDataAlgoPerf_t *)malloc(bwd_data_req_count * sizeof(cudnnConvolutionBwdDataAlgoPerf_t));
	checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithm(cudnn_handle,
															filter_desc, output_tensor, conv_desc, input_tensor, 
															bwd_data_req_count, &bwd_data_ret_count, bwd_data_perf));

	output_size.N = output_batch_size, output_size.C = output_channels, output_size.H = output_h, output_size.W = output_w;
	
}

void ConvLayerParams::allocateSpace(curandGenerator_t curand_gen, cudnnDataType_t data_type, size_t data_type_size, 
									float std_dev, size_t &free_bytes) {

	int kernel_size = C_out * C_in * filter_h * filter_w;
	if (kernel_size % 2 != 0)
		kernel_size += 1;
	checkCudaErrors(cudaMalloc(&W, 
								kernel_size * data_type_size));
	checkCudaErrors(cudaMalloc(&b, C_out * data_type_size));

	checkCudaErrors(cudaMalloc(&dW, 
								kernel_size * data_type_size));
	checkCudaErrors(cudaMalloc(&db, C_out * data_type_size));

	if (data_type == CUDNN_DATA_FLOAT) {
		checkCURAND(curandGenerateNormal(curand_gen, (float *)W, kernel_size, 0, std_dev));
		fillValue<float><<<ceil(1.0 * C_out / BW), BW>>>((float *)b, C_out, 0);
	}
	else {
		checkCURAND(curandGenerateNormalDouble(curand_gen, (double *)W, kernel_size, 0, std_dev));
		fillValue<double><<<ceil(1.0 * C_out / BW), BW>>>((double *)b, C_out, 0);
	}

	free_bytes = free_bytes - 2 * (kernel_size + C_out) * data_type_size;

}

void ConvLayerParams::getWorkspaceSize(size_t &workspace_size, size_t &free_bytes, ConvAlgo conv_algo) {
	int min_time;
	int min_index;
	size_t memory;
	size_t max_memory;
	min_time = std::numeric_limits<int>::max();
	min_index = -1;
	if (conv_algo == CONV_ALGO_AUTO) {
		for (int i = 0; i < fwd_ret_count; i++) {
			if (fwd_perf[i].status == CUDNN_STATUS_SUCCESS && fwd_perf[i].memory < free_bytes && fwd_perf[i].time < min_time) {
				min_time = fwd_perf[i].time;
				min_index = i;
			}
		}
		memory = fwd_perf[min_index].memory;
		fwd_algo = fwd_perf[min_index].algo;
		max_memory = memory;
	
		min_time = std::numeric_limits<int>::max();
		min_index = -1;
		for (int i = 0; i < bwd_filter_ret_count; i++) {
			if (bwd_filter_perf[i].status == CUDNN_STATUS_SUCCESS && bwd_filter_perf[i].memory < free_bytes && bwd_filter_perf[i].time < min_time) {
				min_time = bwd_filter_perf[i].time;
				min_index = i;
			}
		}
		memory = bwd_filter_perf[min_index].memory;
		bwd_filter_algo = bwd_filter_perf[min_index].algo;
		// std::cout << "ConvLayerParams: workspace, filter_algo: memory: " << memory << " algo: " << bwd_filter_algo << std::endl;
		max_memory = (memory > max_memory) ? memory : max_memory;
	
		min_time = std::numeric_limits<int>::max();
		min_index = -1;
		for (int i = 0; i < bwd_data_ret_count; i++) {
			if (bwd_data_perf[i].status == CUDNN_STATUS_SUCCESS && bwd_data_perf[i].memory < free_bytes && bwd_data_perf[i].time < min_time) {
				min_time = bwd_data_perf[i].time;
				min_index = i;
			}
		}
		memory = bwd_data_perf[min_index].memory;
		bwd_data_algo = bwd_data_perf[min_index].algo;
		max_memory = (memory > max_memory) ? memory : max_memory;
	}
	else if (conv_algo == CONV_ALGO_MEMORY_OPTIMAL) {
		for (int i = 0; i < fwd_ret_count; i++) {
			if (fwd_perf[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM) {
				if (fwd_perf[i].status == CUDNN_STATUS_SUCCESS && fwd_perf[i].memory < free_bytes) {
					fwd_algo = fwd_perf[i].algo;
					memory = fwd_perf[i].memory;
					max_memory = memory;
					break;
				}
				else {
					std::cout << "workspace: bad_status or out of memory\n";
					exit(0);
				}
			}
		}
		for (int i = 0; i < bwd_filter_ret_count; i++) {
			if (bwd_filter_perf[i].algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1) {
				if (bwd_filter_perf[i].status == CUDNN_STATUS_SUCCESS && bwd_filter_perf[i].memory < free_bytes) {
					bwd_filter_algo = bwd_filter_perf[i].algo;
					// std::cout << "Free bytes " << free_bytes << std::endl;
					// std::cout << "bwd_filter_perf[i].memory " << bwd_filter_perf[i].memory << std::endl;
					memory = bwd_filter_perf[i].memory;
					max_memory = (memory > max_memory) ? memory : max_memory;
					break;
				}
				else {
					std::cout << "workspace: bad_status or out of memory\n";
					exit(0);
				}
			}
		}
		for (int i = 0; i < bwd_data_ret_count; i++) {
			if (bwd_data_perf[i].algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_1) {
				if(bwd_data_perf[i].status == CUDNN_STATUS_SUCCESS && bwd_data_perf[i].memory < free_bytes) {
					bwd_data_algo = bwd_data_perf[i].algo;
					memory = bwd_data_perf[i].memory;
					max_memory = (memory > max_memory) ? memory : max_memory;
					break;
				}
				else {
					std::cout << "workspace: bad_status or out of memory\n";
					exit(0);
				}
			}
		}
	}
	else if (conv_algo == CONV_ALGO_PERFORMANCE_OPTIMAL) {
		if (fwd_perf[0].status == CUDNN_STATUS_SUCCESS && fwd_perf[0].memory < free_bytes) {
			fwd_algo = fwd_perf[0].algo;
			memory = fwd_perf[0].memory;
			max_memory = memory;
		}
		else {
			std::cout << "workspace: bad_status or out of memory\n";
			exit(0);
		}
		if (bwd_filter_perf[0].status == CUDNN_STATUS_SUCCESS && bwd_filter_perf[0].memory < free_bytes) {
			bwd_filter_algo = bwd_filter_perf[0].algo;
			// std::cout << "Free bytes " << free_bytes << std::endl;
			// std::cout << "bwd_filter_perf[i].memory " << bwd_filter_perf[i].memory << std::endl;
			memory = bwd_filter_perf[0].memory;
			max_memory = (memory > max_memory) ? memory : max_memory;
		}
		else {
			std::cout << "workspace: bad_status or out of memory\n";
			exit(0);
		}
		if(bwd_data_perf[0].status == CUDNN_STATUS_SUCCESS && bwd_data_perf[0].memory < free_bytes) {
			bwd_data_algo = bwd_data_perf[0].algo;
			memory = bwd_data_perf[0].memory;
			max_memory = (memory > max_memory) ? memory : max_memory;
		}
		else {
			std::cout << "workspace: bad_status or out of memory\n";
			exit(0);
		}
	}
	workspace_size = max_memory;
}

void FCLayerParams::initializeValues(FCDescriptor *user_params, int batch_size, size_t data_type_size, LayerDimension &output_size) {
	C_in = user_params->input_channels;
	C_out = user_params->output_channels;

	output_size.N = batch_size, output_size.C = C_out, output_size.H = output_size.W = 1;
}

void FCLayerParams::allocateSpace(curandGenerator_t curand_gen, cudnnDataType_t data_type, size_t data_type_size, 
									float std_dev, size_t &free_bytes) {
	int wt_alloc_size = C_in * C_out;
	if (wt_alloc_size % 2 != 0)
		wt_alloc_size += 1;
	checkCudaErrors(cudaMalloc(&W, wt_alloc_size * data_type_size));
	checkCudaErrors(cudaMalloc(&b, C_out * data_type_size));
	checkCudaErrors(cudaMalloc(&dW, wt_alloc_size * data_type_size));
	checkCudaErrors(cudaMalloc(&db, C_out * data_type_size));

	if (data_type == CUDNN_DATA_FLOAT) {
		checkCURAND(curandGenerateNormal(curand_gen, (float *)W, wt_alloc_size, 0, std_dev));
		fillValue<float><<<ceil(1.0 * C_out / BW), BW>>>((float *)b, C_out, 0);
	}
	else if (data_type == CUDNN_DATA_DOUBLE) {
		checkCURAND(curandGenerateNormalDouble(curand_gen, (double *)W, wt_alloc_size, 0, std_dev));
		fillValue<double><<<ceil(1.0 * C_out / BW), BW>>>((double *)b, C_out, 0);
	}
	free_bytes = free_bytes - 2 * (C_in * C_out + C_out) * data_type_size;
}

void DropoutLayerParams::initializeValues(cudnnHandle_t cudnn_handle, DropoutDescriptor *user_params, cudnnDataType_t data_type, int batch_size,
										 cudnnTensorFormat_t tensor_format, LayerDimension &output_size) {
	checkCUDNN(cudnnCreateDropoutDescriptor(&dropout_desc));
	checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));

	checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, 
										batch_size, user_params->channels, user_params->h, user_params->w));

	checkCUDNN(cudnnDropoutGetStatesSize(cudnn_handle, &state_size));

	checkCUDNN(cudnnDropoutGetReserveSpaceSize(input_tensor, &reserved_space_size));
	
	output_size.N = batch_size, output_size.C = user_params->channels, output_size.H = user_params->h, output_size.W = user_params->w;

}

void DropoutLayerParams::allocateSpace(size_t &free_bytes, cudnnHandle_t cudnn_handle, DropoutDescriptor *user_params, long long seed) {
	checkCudaErrors(cudaMalloc(&state, state_size));
	checkCudaErrors(cudaMalloc(&reserved_space, reserved_space_size));
	checkCUDNN(cudnnSetDropoutDescriptor(dropout_desc, cudnn_handle, user_params->dropout_value, state, state_size, seed));

	free_bytes = free_bytes - (state_size + reserved_space_size);
}

void BatchNormLayerParams::initializeValues(BatchNormDescriptor *user_params, cudnnDataType_t data_type, cudnnTensorFormat_t tensor_format, 
							int batch_size, LayerDimension &output_size) {
	checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&sbmv_desc));
	c = user_params->channels, h = user_params->h, w = user_params->w;
	if (user_params->mode == BATCHNORM_PER_ACTIVATION) {
		mode = CUDNN_BATCHNORM_PER_ACTIVATION;
		checkCUDNN(cudnnSetTensor4dDescriptor(sbmv_desc, tensor_format, data_type,
												1, user_params->channels, user_params->h, user_params->w));
		sbmv_size = c * h * w;
	}
	else if (user_params->mode == BATCHNORM_SPATIAL) {
		mode = CUDNN_BATCHNORM_SPATIAL;
		checkCUDNN(cudnnSetTensor4dDescriptor(sbmv_desc, tensor_format, data_type,
												1, user_params->channels, 1, 1));
		sbmv_size = c;
	}

	checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, 
										batch_size, user_params->channels, user_params->h, user_params->w));
	
	factor = user_params->factor;
	epsilon = user_params->epsilon;

	output_size.N = batch_size, output_size.C = user_params->channels, output_size.H = user_params->h, output_size.W = user_params->w;
}

void BatchNormLayerParams::allocateSpace(cudnnDataType_t data_type, size_t data_type_size, size_t &free_bytes) {
	
	size_t allocation_size;
	if (mode == CUDNN_BATCHNORM_PER_ACTIVATION)
		allocation_size = c * h * w;
	else
		allocation_size = c;

	allocation_size *= data_type_size;

	checkCudaErrors(cudaMalloc(&scale, allocation_size));
	checkCudaErrors(cudaMalloc(&bias, allocation_size));
	checkCudaErrors(cudaMalloc(&dscale, allocation_size));
	checkCudaErrors(cudaMalloc(&dbias, allocation_size));

	checkCudaErrors(cudaMalloc(&running_mean, allocation_size));
	checkCudaErrors(cudaMalloc(&running_variance, allocation_size));

	checkCudaErrors(cudaMalloc(&result_save_mean, allocation_size));
	checkCudaErrors(cudaMalloc(&result_save_inv_var, allocation_size));

	int num_elements = allocation_size / data_type_size;
	if (data_type == CUDNN_DATA_FLOAT) {
		fillValue<float><<<ceil(1.0 * num_elements / BW), BW>>>((float *)scale, num_elements, 1);
		fillValue<float><<<ceil(1.0 * num_elements / BW), BW>>>((float *)bias, num_elements, 1);
	}
	else if (data_type == CUDNN_DATA_DOUBLE) {
		fillValue<double><<<ceil(1.0 * num_elements / BW), BW>>>((double *)scale, num_elements, 1);
		fillValue<double><<<ceil(1.0 * num_elements / BW), BW>>>((double *)bias, num_elements, 1);
	}
	free_bytes = free_bytes - 6 * allocation_size;

}

void PoolingLayerParams::initializeValues(PoolingDescriptor *user_params, cudnnDataType_t data_type, cudnnTensorFormat_t tensor_format, 
							int batch_size, LayerDimension &output_size) {
	checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&output_tensor));

	checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, 
										batch_size, user_params->input_channels, user_params->input_h, user_params->input_w));
	

	checkCUDNN(cudnnCreatePoolingDescriptor(&pool_desc));

	cudnnPoolingMode_t mode;
	if (user_params->mode == POOLING_MAX)
		mode = CUDNN_POOLING_MAX;
	else if (user_params->mode == POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
		mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
	else if (user_params->mode == POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
		mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

	checkCUDNN(cudnnSetPooling2dDescriptor(pool_desc, mode, CUDNN_PROPAGATE_NAN,
											user_params->kernel_h, user_params->kernel_w,
											user_params->pad_h, user_params->pad_w,
											user_params->stride_y, user_params->stride_x));


	int output_batch_size, output_channels, output_h, output_w;
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(pool_desc, input_tensor, 
											&output_batch_size, &output_channels, &output_h, &output_w));

	checkCUDNN(cudnnSetTensor4dDescriptor(output_tensor, tensor_format, data_type, 
										output_batch_size, output_channels, output_h, output_w));

	output_size.N = output_batch_size, output_size.C = output_channels, output_size.H = output_h, output_size.W = output_w;

}

void PoolingLayerParams::allocateSpace(size_t &free_bytes) {

}

void ActivationLayerParams::initializeValues(ActivationDescriptor *user_params, cudnnDataType_t data_type,
											cudnnTensorFormat_t tensor_format, int batch_size, LayerDimension &output_size) {
	checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));

	checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, 
										batch_size, user_params->channels, user_params->h, user_params->w));

	cudnnActivationMode_t mode;
	if (user_params->mode == SIGMOID)
		mode = CUDNN_ACTIVATION_SIGMOID;
	else if (user_params->mode == RELU)
		mode = CUDNN_ACTIVATION_RELU;
	else if (user_params->mode == TANH)
		mode = CUDNN_ACTIVATION_TANH;
	else if (user_params->mode == CLIPPED_RELU)
		mode = CUDNN_ACTIVATION_CLIPPED_RELU;
	else if (user_params->mode == ELU)
		mode = CUDNN_ACTIVATION_ELU;

	checkCUDNN(cudnnCreateActivationDescriptor(&actv_desc));
	checkCUDNN(cudnnSetActivationDescriptor(actv_desc, mode, CUDNN_PROPAGATE_NAN, user_params->coef));

	output_size.N = batch_size, output_size.C = user_params->channels, output_size.H = user_params->h, output_size.W = user_params->w;
}

void ActivationLayerParams::allocateSpace(size_t &free_bytes) {
	
}

void SoftmaxLayerParams::initializeValues(SoftmaxDescriptor *user_params, cudnnDataType_t data_type,
											cudnnTensorFormat_t tensor_format, int batch_size, LayerDimension &output_size) {
	if (user_params->algo == SOFTMAX_FAST)
		algo = CUDNN_SOFTMAX_FAST;
	else if (user_params->algo == SOFTMAX_ACCURATE)
		algo = CUDNN_SOFTMAX_ACCURATE;

	if (user_params->mode == SOFTMAX_MODE_INSTANCE)
		mode = CUDNN_SOFTMAX_MODE_INSTANCE;
	else if (user_params->mode == SOFTMAX_MODE_CHANNEL) {
		mode = CUDNN_SOFTMAX_MODE_CHANNEL;
	}

	checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, 
										batch_size, user_params->channels, user_params->h, user_params->w));

	output_size.N = batch_size, output_size.C = user_params->channels, output_size.H = user_params->h, output_size.W = user_params->w;	
}

void SoftmaxLayerParams::allocateSpace(size_t &free_bytes) {

}