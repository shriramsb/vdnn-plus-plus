#include "neural_net.h"

template <typename T>
__global__ void softmaxLossBackProp(int *y, T *SO, T *dSO, int batch_size, int output_size, float eps) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= batch_size)
		return;
	int cur_class = static_cast<int>(y[i]);
	dSO[i * output_size + cur_class] = -1 / (SO[i * output_size + cur_class] * batch_size + eps);
}

template <typename T>
__global__ void computeSoftmaxLoss(T *O, int *y, float *loss, int batch_size, int num_classes, float eps) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= batch_size)
		return;

	loss[i] = -logf(O[i * num_classes + y[i]] + eps);
}

template <typename T>
__global__ void inferClass(T *O, int *pred_y, int batch_size, int num_classes) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= batch_size)
		return;

	T max = O[i * num_classes];
	int index = 0;
	for (int j = 1; j < num_classes; j++) {
		if (O[i * num_classes + j] > max) {
			max = O[i * num_classes + j];
			index = j;
		}
	}
	pred_y[i] = index;
}



float NeuralNet::computeLoss() {
	if (layer_type[num_layers - 1] == SOFTMAX) {
		if (data_type == CUDNN_DATA_FLOAT)
			computeSoftmaxLoss<float><<<ceil(1.0 * batch_size / BW), BW>>>((float *)layer_input[num_layers], this->y, loss, batch_size, num_classes, softmax_eps);
		else if (data_type == CUDNN_DATA_DOUBLE)
			computeSoftmaxLoss<double><<<ceil(1.0 * batch_size / BW), BW>>>((double *)layer_input[num_layers], this->y, loss, batch_size, num_classes, softmax_eps);
	}
	checkCudaErrors(cudaMemcpy(h_loss, loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
	float total_loss = 0.0;
	for (int i = 0; i < batch_size; i++)
		total_loss += h_loss[i];
	return total_loss / batch_size;
}

void NeuralNet::compareOutputCorrect(int *correct_count, int *y) {
	*correct_count = 0;

	if (data_type == CUDNN_DATA_FLOAT) {
		float *typecast_O = (float *)layer_input[num_layers - 1];
		inferClass<float><<<ceil(1.0 * batch_size / BW), BW>>>(typecast_O, pred_y, batch_size, num_classes);
		checkCudaErrors(cudaMemcpy(h_pred_y, pred_y, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
		for (int i = 0; i < batch_size; i++) {
			if (h_pred_y[i] == y[i])
				*correct_count = *correct_count + 1;
		}
	}
	else if (data_type == CUDNN_DATA_DOUBLE) {
		double *typecast_O = (double *)layer_input[num_layers - 1];
		inferClass<double><<<ceil(1.0 * batch_size / BW), BW>>>(typecast_O, pred_y, batch_size, num_classes);
		checkCudaErrors(cudaMemcpy(h_pred_y, pred_y, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
		for (int i = 0; i < batch_size; i++) {
			if (h_pred_y[i] == y[i])
				*correct_count = *correct_count + 1;
		}
	}
}




NeuralNet::NeuralNet(std::vector<LayerSpecifier> &layers, DataType data_type, int batch_size, TensorFormat tensor_format, 
						long long dropout_seed, float softmax_eps, float init_std_dev, ConvAlgo conv_algo) {
	// create handle
	checkCUDNN(cudnnCreate(&cudnn_handle));
	checkCUBLAS(cublasCreate(&cublas_handle));
	checkCURAND(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));

	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
	size_t init_free_bytes = free_bytes;
	std::cout << "Free bytes at start: " << free_bytes << std::endl;

	if (data_type == DATA_FLOAT) {
		this->data_type = CUDNN_DATA_FLOAT;
		data_type_size = sizeof(float);
	}

	else if (data_type == DATA_DOUBLE) {
		this->data_type = CUDNN_DATA_DOUBLE;
		data_type_size = sizeof(double);
	}

	if (tensor_format == TENSOR_NCHW)
		this->tensor_format = CUDNN_TENSOR_NCHW;
	else if (tensor_format == TENSOR_NHWC)
		this->tensor_format = CUDNN_TENSOR_NHWC;

	this->batch_size = batch_size;
	this->softmax_eps = softmax_eps;
	this->init_std_dev = init_std_dev;
	this->conv_algo = conv_algo;

	num_layers = layers.size();
	// allocation of space for input to each layer
	layer_input = (void **)malloc((num_layers + 1) * sizeof(void *));
	dlayer_input = (void **)malloc(2 * sizeof(void *));
	params = (void **)malloc(num_layers * sizeof(void *));

	LayerDimension prev_output_size;
	LayerDimension current_output_size;
	for (int i = 0; i < num_layers; i++) {
		layer_type.push_back(layers[i].type);
		if (layers[i].type == CONV) {
			ConvDescriptor *user_params = (ConvDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(ConvLayerParams));
			((ConvLayerParams *)params[i])->initializeValues(cudnn_handle, user_params, this->data_type, batch_size, this->tensor_format, 
																data_type_size, current_output_size);
		}
		else if (layers[i].type == FULLY_CONNECTED) {
			FCDescriptor *user_params = (FCDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(FCLayerParams));
			((FCLayerParams *)params[i])->initializeValues(user_params, batch_size, data_type_size, current_output_size);
		}
		else if (layers[i].type == DROPOUT) {
			DropoutDescriptor *user_params = (DropoutDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(DropoutLayerParams));
			((DropoutLayerParams *)params[i])->initializeValues(cudnn_handle, user_params, this->data_type, batch_size, 
																this->tensor_format, current_output_size);
			
		}

		else if (layers[i].type == BATCHNORM) {
			BatchNormDescriptor *user_params = (BatchNormDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(BatchNormLayerParams));
			((BatchNormLayerParams *)params[i])->initializeValues(user_params, this->data_type, this->tensor_format, batch_size, current_output_size);
			
		}
		else if (layers[i].type == POOLING) {
			PoolingDescriptor *user_params = (PoolingDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(BatchNormLayerParams));

			((PoolingLayerParams *)params[i])->initializeValues(user_params, this->data_type, this->tensor_format, 
																batch_size, current_output_size);
		}

		else if (layers[i].type == ACTV) {
			ActivationDescriptor *user_params = (ActivationDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(ActivationLayerParams));
			((ActivationLayerParams *)params[i])->initializeValues(user_params, this->data_type, this->tensor_format, 
																	batch_size, current_output_size);
		}

		else if (layers[i].type == SOFTMAX) {
			SoftmaxDescriptor *user_params = (SoftmaxDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(SoftmaxLayerParams));
			((SoftmaxLayerParams *)params[i])->initializeValues(user_params, this->data_type, this->tensor_format, 
																batch_size, current_output_size);
			// std::cout << current_output_size.N << ' ' << current_output_size.C << current_output_size.H << current_output_size.W << std::endl;
		}
		if (i == 0) {
			prev_output_size = current_output_size;
		}
		// incomplete - have to check flatten and check exact dimension
		// else if (current_output_size.getTotalSize() != prev_output_size.getTotalSize()) {
		// 	std::cout << "Layer " << i << " output and next layer's input size mismatch\n";
		// 	exit(0);
		// }
	}

	bool actv_layer = false;
	size_t max_input_size = 0;
	for (int i = 0; i < num_layers; i++) {
		size_t input_size;
		if (layers[i].type == CONV) {
			ConvDescriptor *user_params = (ConvDescriptor *)layers[i].params;
			((ConvLayerParams *)params[i])->allocateSpace(curand_gen, this->data_type, data_type_size, init_std_dev, free_bytes);
			input_size = batch_size * user_params->input_channels * user_params->input_h * user_params->input_w;
			max_input_size = (input_size > max_input_size) ? input_size : max_input_size;
			if (i == 0) {
				input_channels = user_params->input_channels;
				input_h = user_params->input_h;
				input_w = user_params->input_w;
			}
		}
		else if (layers[i].type == FULLY_CONNECTED) {
			FCDescriptor *user_params = (FCDescriptor *)layers[i].params;
			((FCLayerParams *)params[i])->allocateSpace(curand_gen, this->data_type, data_type_size, init_std_dev, free_bytes);
			input_size = batch_size * user_params->input_channels;
			max_input_size = (input_size > max_input_size) ? input_size : max_input_size;
			if (i == 0) {
				input_channels = user_params->input_channels;
				input_h = 1;
				input_w = 1;
			}
		}
		else if (layers[i].type == DROPOUT) {
			DropoutDescriptor *user_params = (DropoutDescriptor *)layers[i].params;
			((DropoutLayerParams *)params[i])->allocateSpace(free_bytes, cudnn_handle, user_params, dropout_seed);
			input_size = batch_size * user_params->channels * user_params->h * user_params->w;
			max_input_size = (input_size > max_input_size) ? input_size : max_input_size;
			if (i == 0) {
				input_channels = user_params->channels;
				input_h = user_params->h;
				input_w = user_params->w;
			}
		}
		else if (layers[i].type == BATCHNORM) {
			BatchNormDescriptor *user_params = (BatchNormDescriptor *)layers[i].params;
			((BatchNormLayerParams *)params[i])->allocateSpace(this->data_type, data_type_size, free_bytes);
			input_size = batch_size * user_params->channels * user_params->h * user_params->w;
			max_input_size = (input_size > max_input_size) ? input_size : max_input_size;
			if (i == 0) {
				input_channels = user_params->channels;
				input_h = user_params->h;
				input_w = user_params->w;
			}
		}
		else if (layers[i].type == POOLING) {
			PoolingDescriptor *user_params = (PoolingDescriptor *)layers[i].params;
			((PoolingLayerParams *)params[i])->allocateSpace(free_bytes);
			input_size = batch_size * user_params->input_channels * user_params->input_h * user_params->input_w;
			max_input_size = (input_size > max_input_size) ? input_size : max_input_size;
			if (i == 0) {
				input_channels = user_params->input_channels;
				input_h = user_params->input_h;
				input_w = user_params->input_w;
			}
		}
		else if (layers[i].type == ACTV) {
			ActivationDescriptor *user_params = (ActivationDescriptor *)layers[i].params;
			((ActivationLayerParams *)params[i])->allocateSpace(free_bytes);
			input_size = batch_size * user_params->channels * user_params->h * user_params->w;
			max_input_size = (input_size > max_input_size) ? input_size : max_input_size;
			if (i == 0) {
				input_channels = user_params->channels;
				input_h = user_params->h;
				input_w = user_params->w;
			}
		}
		else if (layers[i].type == SOFTMAX) {
			SoftmaxDescriptor *user_params = (SoftmaxDescriptor *)layers[i].params;
			((SoftmaxLayerParams *)params[i])->allocateSpace(free_bytes);
			input_size = batch_size * user_params->channels * user_params->h * user_params->w;
			max_input_size = (input_size > max_input_size) ? input_size : max_input_size;

			// assuming this is last layer, allocate for next layer as well
			// checkCudaErrors(cudaMalloc(&layer_input[i + 1], input_size * data_type_size));
			// checkCudaErrors(cudaMalloc(&dlayer_input[i + 1], input_size * data_type_size));
			
			if (i == 0) {
				input_channels = user_params->channels;
				input_h = user_params->h;
				input_w = user_params->w;
			}
			if (i == num_layers - 1) {
				num_classes = user_params->channels;
			}
		}
		if (actv_layer == false) {
			checkCudaErrors(cudaMalloc(&layer_input[i], input_size * data_type_size));
			// checkCudaErrors(cudaMalloc(&dlayer_input[i], input_size * data_type_size));
		}
		else {
			layer_input[i] = layer_input[i - 1];
			// dlayer_input[i] = dlayer_input[i - 1];
		}
		if (layers[i].type == ACTV)
			actv_layer = true;
		else
			actv_layer = false;
		if (layers[i].type == SOFTMAX) {
			// do not allocate space for output of softmax
			layer_input[i + 1] = layer_input[i];
			// dlayer_input[i + 1] = dlayer_input[i];
		}
	}
	checkCudaErrors(cudaMalloc((void **)&y, batch_size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&pred_y, batch_size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&loss, batch_size * sizeof(float)));
	checkCudaErrors(cudaMalloc(&one_vec, batch_size * data_type_size));
	if (this->data_type == CUDNN_DATA_FLOAT)
		fillValue<float><<<ceil(1.0 * batch_size / BW), BW>>>((float *)one_vec, batch_size, 1);
	else
		fillValue<double><<<ceil(1.0 * batch_size / BW), BW>>>((double *)one_vec, batch_size, 1);
	
	checkCudaErrors(cudaMalloc(&dlayer_input[0], max_input_size * data_type_size));
	checkCudaErrors(cudaMalloc(&dlayer_input[1], max_input_size * data_type_size));

	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));

	// allocate space for workspace and also keep track of algo
	size_t cur_workspace_size;
	workspace_size = 0;
	for (int i = 0; i < num_layers; i++) {
		if (layers[i].type == CONV) {
			((ConvLayerParams *)params[i])->getWorkspaceSize(cur_workspace_size, free_bytes, conv_algo);
			if (cur_workspace_size > workspace_size)
				workspace_size = cur_workspace_size;
		}
	}

	checkCudaErrors(cudaMalloc(&workspace, workspace_size));
	free_bytes = free_bytes - workspace_size;

	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
	std::cout << "Free bytes just before end of NeuralNet: " << free_bytes << std::endl;
	{
		// int n;
		// std::cout << "Waiting..\n";
		// std::cin >> n;
	}

	// if total consume crosses 3.4 GB, stop
	std::cout << "Total consume: " << init_free_bytes - free_bytes << std::endl;
	std::cout << "Total consume(MB): " << (init_free_bytes - free_bytes) / (1.0 * 1024 * 1024) << std::endl;
	if ((total_bytes - free_bytes) > 3400l * 1024 * 1024) {
		std::cout << "Crosses 3.4GB\n";
		exit(0);
	}

	checkCudaErrors(cudaMallocHost((void **)&h_loss, batch_size * sizeof(float)));
	checkCudaErrors(cudaMallocHost((void **)&h_pred_y, batch_size * sizeof(int)));

	
}

void NeuralNet::getLoss(void *X, int *y, bool train, int *correct_count, float *scalar_loss) {

	checkCudaErrors(cudaMemcpy(layer_input[0], X, batch_size * input_channels * input_h * input_w * data_type_size, cudaMemcpyHostToDevice));
	if (train == true) {
		checkCudaErrors(cudaMemcpy(this->y, y, batch_size * data_type_size, cudaMemcpyHostToDevice));
	}
	float alpha = 1.0, beta = 0.0;
	float Salpha = 1.0, Sbeta = 0.0;
	double Dalpha = 1.0, Dbeta = 0.0;

	float milli = 0;
	

	
	// forward propagate
	for (int i = 0; i < num_layers; i++) {
		if (layer_type[i] == CONV) {
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];
			checkCUDNN(cudnnConvolutionForward(cudnn_handle, &alpha, 
												cur_params->input_tensor, layer_input[i],
												cur_params->filter_desc, cur_params->W,
												cur_params->conv_desc, cur_params->fwd_algo,
												workspace, workspace_size,
												&beta,
												cur_params->output_tensor, layer_input[i + 1]));


			checkCUDNN(cudnnAddTensor(cudnn_handle, &alpha, 
										cur_params->bias_desc, cur_params->b, 
										&alpha,
										cur_params->output_tensor, layer_input[i + 1]));
		}

		else if (layer_type[i] == FULLY_CONNECTED) {
			FCLayerParams *cur_params = (FCLayerParams *)params[i];
			if (data_type == CUDNN_DATA_FLOAT) {
				checkCUBLAS(cublasSgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										cur_params->C_out, batch_size, cur_params->C_in,
										&Salpha,
										(float *)cur_params->W, cur_params->C_out,
										(float *)layer_input[i], cur_params->C_in,
										&Sbeta,
										(float *)layer_input[i + 1], cur_params->C_out));
				checkCUBLAS(cublasSgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										cur_params->C_out, batch_size, 1,
										&Salpha,
										(float *)cur_params->b, cur_params->C_out,
										(float *)one_vec, 1,
										&Salpha,
										(float *)layer_input[i + 1], cur_params->C_out));
			}
			else if (data_type == CUDNN_DATA_DOUBLE) {
				checkCUBLAS(cublasDgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										cur_params->C_out, batch_size, cur_params->C_in,
										&Dalpha,
										(double *)cur_params->W, cur_params->C_out,
										(double *)layer_input[i], cur_params->C_in,
										&Dbeta,
										(double *)layer_input[i + 1], cur_params->C_out));
				checkCUBLAS(cublasDgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										cur_params->C_out, batch_size, 1,
										&Dalpha,
										(double *)cur_params->b, cur_params->C_out,
										(double *)one_vec, 1,
										&Dalpha,
										(double *)layer_input[i + 1], cur_params->C_out));
			}
		}
		else if (layer_type[i] == DROPOUT) {
			DropoutLayerParams *cur_params = (DropoutLayerParams *)params[i];
			checkCUDNN(cudnnDropoutForward(cudnn_handle, cur_params->dropout_desc,
											cur_params->input_tensor, layer_input[i],
											cur_params->input_tensor, layer_input[i + 1],
											cur_params->reserved_space,
											cur_params->reserved_space_size));
		}
		else if (layer_type[i] == BATCHNORM) {
			BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];
			if (train == true) {
				checkCUDNN(cudnnBatchNormalizationForwardTraining(cudnn_handle, cur_params->mode,
																	&alpha, &beta,
																	cur_params->input_tensor, layer_input[i],
																	cur_params->input_tensor, layer_input[i + 1],
																	cur_params->sbmv_desc,
																	cur_params->scale, cur_params->bias,
																	cur_params->factor,
																	cur_params->running_mean, cur_params->running_variance,
																	cur_params->epsilon,
																	cur_params->result_save_mean, cur_params->result_save_inv_var));
			}
			else {
				checkCUDNN(cudnnBatchNormalizationForwardInference(cudnn_handle, cur_params->mode,
																	&alpha, &beta,
																	cur_params->input_tensor, layer_input[i],
																	cur_params->input_tensor, layer_input[i + 1],
																	cur_params->sbmv_desc,
																	cur_params->scale, cur_params->bias,
																	cur_params->running_mean, cur_params->running_variance,
																	cur_params->epsilon));
			}
		}
		else if (layer_type[i] == POOLING) {
			PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
			checkCUDNN(cudnnPoolingForward(cudnn_handle, cur_params->pool_desc,
											&alpha,
											cur_params->input_tensor, layer_input[i],
											&beta,
											cur_params->output_tensor, layer_input[i + 1]));
		}
		else if (layer_type[i] == ACTV) {
			ActivationLayerParams *cur_params = (ActivationLayerParams *)params[i];
			checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
												&alpha,
												cur_params->input_tensor, layer_input[i],
												&beta,
												cur_params->input_tensor, layer_input[i + 1]));
		}
		else if (layer_type[i] == SOFTMAX) {
			
			if (train == true) {
				SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
						checkCUDNN(cudnnSoftmaxForward(cudnn_handle, cur_params->algo, cur_params->mode,
														&alpha,
														cur_params->input_tensor, layer_input[i],
														&beta,
														cur_params->input_tensor, layer_input[i + 1]));
			}
		}
	}

	if (train == false) {
		compareOutputCorrect(correct_count, y);
		return;
	}
	*scalar_loss = computeLoss();

	int cur_derv = 0;
	int next_derv = 1;

	if (layer_type[num_layers - 1] == SOFTMAX) {
		// SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[num_layers - 1];
		if (data_type == CUDNN_DATA_FLOAT) {
			checkCudaErrors(cudaMemset(dlayer_input[cur_derv], 0, batch_size * num_classes * sizeof(float)));
			softmaxLossBackProp<float><<<ceil(1.0 * batch_size / BW), BW>>>(this->y, (float *)layer_input[num_layers], 
																			(float *)dlayer_input[cur_derv], batch_size, num_classes, softmax_eps);
		}
		else if (data_type == CUDNN_DATA_DOUBLE) {
			checkCudaErrors(cudaMemset(dlayer_input[cur_derv], 0, batch_size * num_classes * sizeof(double)));
			softmaxLossBackProp<double><<<ceil(1.0 * batch_size / BW), BW>>>(this->y, (double *)layer_input[num_layers], 
																			(double *)dlayer_input[cur_derv], batch_size, num_classes, softmax_eps);
		}
	}
	for (int i = num_layers - 1; i >= 0; i--) {

		if (layer_type[i] == CONV) {
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];
			checkCUDNN(cudnnConvolutionBackwardBias(cudnn_handle, &alpha,
													cur_params->output_tensor, dlayer_input[cur_derv],
													&beta,
													cur_params->bias_desc, cur_params->db));

			// std::cout << "neural_net: backward conv i:" << i << std::endl; 
			checkCUDNN(cudnnConvolutionBackwardFilter(cudnn_handle, &alpha,
														cur_params->input_tensor, layer_input[i],
														cur_params->output_tensor, dlayer_input[cur_derv],
														cur_params->conv_desc, cur_params->bwd_filter_algo,
														workspace, workspace_size,
														&beta, 
														cur_params->filter_desc,
														cur_params->dW));

			if (i > 0)
				checkCUDNN(cudnnConvolutionBackwardData(cudnn_handle, &alpha,
														cur_params->filter_desc, cur_params->W,
														cur_params->output_tensor, dlayer_input[cur_derv],
														cur_params->conv_desc, cur_params->bwd_data_algo,
														workspace, workspace_size,
														&beta,
														cur_params->input_tensor, dlayer_input[next_derv]));
		}

		else if (layer_type[i] == FULLY_CONNECTED) {
			FCLayerParams *cur_params = (FCLayerParams *)params[i];
			if (data_type == CUDNN_DATA_FLOAT) {
				// bias backward
				checkCUBLAS(cublasSgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										cur_params->C_out, 1, batch_size,
										&Salpha,
										(float *)dlayer_input[cur_derv], cur_params->C_out,
										(float *)one_vec, batch_size,
										&Sbeta,
										(float *)cur_params->db, cur_params->C_out));

				// weight backward
				checkCUBLAS(cublasSgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_T,
										cur_params->C_out, cur_params->C_in, batch_size,
										&Salpha,
										(float *)dlayer_input[cur_derv], cur_params->C_out,
										(float *)layer_input[i], cur_params->C_in,
										&Sbeta,
										(float *)cur_params->dW, cur_params->C_out));

				// data backward
				if (i > 0)
					checkCUBLAS(cublasSgemm(cublas_handle,
											CUBLAS_OP_T, CUBLAS_OP_N,
											cur_params->C_in, batch_size, cur_params->C_out,
											&Salpha,
											(float *)cur_params->W, cur_params->C_out,
											(float *)dlayer_input[cur_derv], cur_params->C_out,
											&Sbeta,
											(float *)dlayer_input[next_derv], cur_params->C_in));
			}

			else if (data_type == CUDNN_DATA_DOUBLE) {
				// bias backward
				checkCUBLAS(cublasDgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										cur_params->C_out, 1, batch_size,
										&Dalpha,
										(double *)dlayer_input[cur_derv], cur_params->C_out,
										(double *)one_vec, batch_size,
										&Dbeta,
										(double *)cur_params->db, cur_params->C_out));

				// weight backward
				checkCUBLAS(cublasDgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_T,
										cur_params->C_out, cur_params->C_in, batch_size,
										&Dalpha,
										(double *)dlayer_input[cur_derv], cur_params->C_out,
										(double *)layer_input[i], cur_params->C_in,
										&Dbeta,
										(double *)cur_params->dW, cur_params->C_out));

				// data backward
				if (i > 0)
					checkCUBLAS(cublasDgemm(cublas_handle,
											CUBLAS_OP_T, CUBLAS_OP_N,
											cur_params->C_in, batch_size, cur_params->C_out,
											&Dalpha,
											(double *)cur_params->W, cur_params->C_out,
											(double *)dlayer_input[cur_derv], cur_params->C_out,
											&Dbeta,
											(double *)dlayer_input[next_derv], cur_params->C_in));
			}
		}

		else if (layer_type[i] == DROPOUT) {
			DropoutLayerParams *cur_params = (DropoutLayerParams *)params[i];
			checkCUDNN(cudnnDropoutBackward(cudnn_handle, cur_params->dropout_desc,
											cur_params->input_tensor, dlayer_input[cur_derv],
											cur_params->input_tensor, dlayer_input[next_derv],
											cur_params->reserved_space, cur_params->reserved_space_size));
		}

		else if (layer_type[i] == BATCHNORM) {
			BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];
			checkCUDNN(cudnnBatchNormalizationBackward(cudnn_handle, cur_params->mode,
														&alpha, &beta,
														&alpha, &beta,
														cur_params->input_tensor, layer_input[i],
														cur_params->input_tensor, dlayer_input[cur_derv],
														cur_params->input_tensor, dlayer_input[next_derv],
														cur_params->sbmv_desc, cur_params->scale,
														cur_params->dscale, cur_params->dbias,
														cur_params->epsilon,
														cur_params->result_save_mean, cur_params->result_save_inv_var));
		}

		else if (layer_type[i] == POOLING) {
			PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
			checkCUDNN(cudnnPoolingBackward(cudnn_handle, cur_params->pool_desc, &alpha, 
											cur_params->output_tensor, layer_input[i + 1],
											cur_params->output_tensor, dlayer_input[cur_derv],
											cur_params->input_tensor, layer_input[i],
											&beta,
											cur_params->input_tensor, dlayer_input[next_derv]));
		}

		else if (layer_type[i] == ACTV) {
			ActivationLayerParams *cur_params = (ActivationLayerParams *)params[i];
			checkCUDNN(cudnnActivationBackward(cudnn_handle, cur_params->actv_desc, &alpha,
												cur_params->input_tensor, layer_input[i + 1],
												cur_params->input_tensor, dlayer_input[cur_derv],
												cur_params->input_tensor, layer_input[i],
												&beta,
												cur_params->input_tensor, dlayer_input[next_derv]));
		}

		else if (layer_type[i] == SOFTMAX) {
			SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
			checkCUDNN(cudnnSoftmaxBackward(cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
											cur_params->input_tensor, layer_input[i + 1],
											cur_params->input_tensor, dlayer_input[cur_derv],
											&beta,
											cur_params->input_tensor, dlayer_input[next_derv]));
		}
		int temp = cur_derv;
		cur_derv = next_derv;
		next_derv = temp;
	}
	checkCudaErrors(cudaDeviceSynchronize());

	
	// exit(0);
}

