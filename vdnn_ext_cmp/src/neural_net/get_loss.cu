#include "neural_net.h"
#include <time.h>

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

void NeuralNet::getLoss(void *X, int *y, double learning_rate, bool train, int *correct_count, float *loss) {
	std::vector<float> t1, t2;
	this->getLoss(X, y, learning_rate, t1, t2, train, correct_count, loss);
}

void NeuralNet::getLoss(void *X, int *y, double learning_rate, std::vector<float> &fwd_vdnn_lag, std::vector<float> &bwd_vdnn_lag, bool train, int *correct_count, float *scalar_loss) {

	CnmemSpace space_tracker(free_bytes);
	// std::cout << "here\n";
	// std::cout << "Free bytes: " << free_bytes << std::endl;
	for (int i = 0; i < num_layers; i++)
		prefetched[i] = false;

	checkCNMEM(cnmemMalloc(&layer_input[0], layer_input_size[0] * data_type_size, NULL));
	space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[0] * data_type_size);
	checkCudaErrors(cudaMemcpy(layer_input[0], X, batch_size * input_channels * input_h * input_w * data_type_size, cudaMemcpyHostToDevice));
	if (train == true) {
		checkCudaErrors(cudaMemcpy(this->y, y, batch_size * data_type_size, cudaMemcpyHostToDevice));
	}
	float alpha = 1.0, beta = 0.0;
	float Salpha = 1.0, Sbeta = 0.0;
	double Dalpha = 1.0, Dbeta = 0.0;

	// forward propagate
	for (int i = 0; i < num_layers; i++) {
		if (train == false && i == num_layers - 1)
			break;
		// ---------------------- vDNN start ----------------------
		size_t cur_workspace_size;
		void *cur_workspace;

		// // offload if required
		// if (i > 0 && to_offload[i] && train == true) {

		// 	checkCudaErrors(cudaMemcpyAsync(h_layer_input[i], layer_input[i], 
		// 									layer_input_size[i] * data_type_size, cudaMemcpyDeviceToHost, stream_memory));
		// 	checkCudaErrors(cudaEventRecord(event_offload_done[i], stream_memory));
		// }

		lockedcnmemMalloc(&layer_input[i + 1], layer_input_size[i + 1] * data_type_size, NULL);
		space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[i + 1] * data_type_size);
		// std::cout << "Free bytes: " << free_bytes << std::endl;
		// ---------------------- vDNN end ------------------------
		// std::cout << "here" << i << std::endl;
		if (layer_type[i] == CONV) {
			// std::cout << "conv\n";
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

			cur_workspace_size = cur_params->fwd_workspace_size;
			lockedcnmemMalloc(&cur_workspace, cur_workspace_size, NULL);			
			// computation
			checkCUDNN(cudnnConvolutionForward(cudnn_handle, &alpha, 
												cur_params->input_tensor, layer_input[i],
												cur_params->filter_desc, cur_params->W,
												cur_params->conv_desc, cur_params->fwd_algo,
												cur_workspace, cur_workspace_size,
												&beta,
												cur_params->output_tensor, layer_input[i + 1]));
			checkCUDNN(cudnnAddTensor(cudnn_handle, &alpha, 
										cur_params->bias_desc, cur_params->b, 
										&alpha,
										cur_params->output_tensor, layer_input[i + 1]));

			// if activation required
			if (cur_params->activation_mode != ACTIVATION_NONE) {
				checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
												&alpha,
												cur_params->output_tensor, layer_input[i + 1],
												&beta,
												cur_params->output_tensor, layer_input[i + 1]));
			}

			space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
			// std::cout << "Free bytes: " << free_bytes << std::endl;
			
		}

		else if (layer_type[i] == FULLY_CONNECTED) {
			// std::cout << "FC\n";
			FCLayerParams *cur_params = (FCLayerParams *)params[i];
			// std::cout << "FChere" << i << std::endl;

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
			if (cur_params->activation_mode != ACTIVATION_NONE) {
				checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
												&alpha,
												cur_params->output_tensor, layer_input[i + 1],
												&beta,
												cur_params->output_tensor, layer_input[i + 1]));
			}
			// std::cout << "FChere" << i << std::endl;
		}
		else if (layer_type[i] == DROPOUT) {
			// std::cout << "Dropout\n";
			DropoutLayerParams *cur_params = (DropoutLayerParams *)params[i];
			checkCUDNN(cudnnDropoutForward(cudnn_handle, cur_params->dropout_desc,
											cur_params->input_tensor, layer_input[i],
											cur_params->input_tensor, layer_input[i + 1],
											cur_params->reserved_space,
											cur_params->reserved_space_size));
		}
		else if (layer_type[i] == BATCHNORM) {
			// std::cout << "Batchnorm\n";
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
			// std::cout << "Pooling\n";
			PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
			checkCUDNN(cudnnPoolingForward(cudnn_handle, cur_params->pool_desc,
											&alpha,
											cur_params->input_tensor, layer_input[i],
											&beta,
											cur_params->output_tensor, layer_input[i + 1]));
		}
		else if (layer_type[i] == ACTV) {
			// std::cout << "Actv\n";
			std::cout << "Panic!! ACTV wrong place\n";
			exit(0);
			ActivationLayerParams *cur_params = (ActivationLayerParams *)params[i];
			checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
												&alpha,
												cur_params->input_tensor, layer_input[i],
												&beta,
												cur_params->input_tensor, layer_input[i + 1]));
		}
		else if (layer_type[i] == SOFTMAX) {
			// std::cout << "Softmax\n";
			std::cout << "Panic!! SOFTMAX wrong place\n";
			exit(0);
			if (train == true) {
				SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
				checkCUDNN(cudnnSoftmaxForward(cudnn_handle, cur_params->algo, cur_params->mode,
												&alpha,
												cur_params->input_tensor, layer_input[i],
												&beta,
												cur_params->input_tensor, layer_input[i + 1]));
			}
		}

		// ---------------------- vDNN start ----------------------
		// synchronization
		// checkCudaErrors(cudaDeviceSynchronize());

		// if next layer is ACTV or SOFTMAX, complete that and come to synchronization
		// the case in above if for ACTV and SOFTMAX never occurs
		if (layer_type[i + 1] == SOFTMAX) {
			i++;
			if (train == true) {
				layer_input[i + 1] = layer_input[i];
				SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
				checkCUDNN(cudnnSoftmaxForward(cudnn_handle, cur_params->algo, cur_params->mode,
												&alpha,
												cur_params->input_tensor, layer_input[i],
												&beta,
												cur_params->input_tensor, layer_input[i + 1]));
			}
			i--;
		}

		// record the point at which computation ends
		checkCudaErrors(cudaEventRecord(event_fwd_compute_done[i], stream_compute));

		if (to_offload[i] && train == true) {
			checkPThreadErrors(pthread_create(&thread_offload_handler[i], NULL, NeuralNet::threadOffloadHandlerHelper, (void *)(&(layer_num[i]))));
			checkPThreadErrors(pthread_detach(thread_offload_handler[i]));
		}

		// struct timespec start_time, end_time;
		checkCudaErrors(cudaStreamSynchronize(stream_compute));

		// if (train && to_offload[i]){
		// 	checkPThreadErrors(pthread_create(&thread_free_layer_input[i], NULL, NeuralNet::threadOffloadHandlerHelper, (void *)(&(layer_num[i]))));
		// 	checkPThreadErrors(pthread_detach(thread_free_layer_input[i]));
		// }
		
		// if (train)
		// 	clock_gettime(CLOCK_MONOTONIC, &start_time);
		
		// checkCudaErrors(cudaStreamSynchronize(stream_memory));
		

		// if (train) {
		// 	clock_gettime(CLOCK_MONOTONIC, &end_time);
		// 	float lag = (end_time.tv_sec - start_time.tv_sec) * 1e3 + (end_time.tv_nsec - start_time.tv_nsec) * 1e-6;
		// 	fwd_vdnn_lag.push_back(lag);
		// }
		if (layer_type[i] == CONV) {
			lockedcnmemFree(cur_workspace, NULL);
			space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
		}

		if (to_offload[i] && train == true) {
			// lockedcnmemFree(layer_input[i], NULL);
			// space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);
		}
		if (train == false) {
			lockedcnmemFree(layer_input[i], NULL);
			space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);
		}

		if (layer_type[i + 1] == ACTV or layer_type[i + 1] == SOFTMAX) {
			i = i + 1;
		}

		// ---------------------- vDNN end ------------------------
	}

	// std::cout << "here" << std::endl;
	if (train == false) {
		compareOutputCorrect(correct_count, y);
		checkCNMEM(cnmemFree(layer_input[num_layers - 1], NULL));
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[num_layers - 1] * data_type_size);
		return;
	}

	struct timespec start_time, end_time;
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	for (int i = 0; i < num_layers; i++) {
		if (to_offload[i])
			checkSemaphoreErrors(sem_wait(&sem_offload_done[i]));
	}
	clock_gettime(CLOCK_MONOTONIC, &end_time);
	float lag = (end_time.tv_sec - start_time.tv_sec) * 1e3 + (end_time.tv_nsec - start_time.tv_nsec) * 1e-6;
	fwd_vdnn_lag.push_back(lag);

	*scalar_loss = computeLoss();

	// ---------------------- vDNN start ----------------------
	checkCNMEM(cnmemMalloc(&dlayer_input[num_layers], batch_size * num_classes * data_type_size, NULL));
	space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[num_layers] * data_type_size);
	// std::cout << "Free bytes: " << free_bytes << std::endl;
	// ---------------------- vDNN end ------------------------
	if (layer_type[num_layers - 1] == SOFTMAX) {
		// SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[num_layers - 1];
		if (data_type == CUDNN_DATA_FLOAT) {
			checkCudaErrors(cudaMemset(dlayer_input[num_layers], 0, batch_size * num_classes * sizeof(float)));
			softmaxLossBackProp<float><<<ceil(1.0 * batch_size / BW), BW>>>(this->y, (float *)layer_input[num_layers], 
																			(float *)dlayer_input[num_layers], batch_size, num_classes, softmax_eps);
		}
		else if (data_type == CUDNN_DATA_DOUBLE) {
			checkCudaErrors(cudaMemset(dlayer_input[num_layers], 0, batch_size * num_classes * sizeof(double)));
			softmaxLossBackProp<double><<<ceil(1.0 * batch_size / BW), BW>>>(this->y, (double *)layer_input[num_layers], 
																			(double *)dlayer_input[num_layers], batch_size, num_classes, softmax_eps);
		}
	}
	for (int i = num_layers - 1; i >= 0; i--) {
		// ---------------------- vDNN start ----------------------
		int cur_filter_workspace_size, cur_data_workspace_size, cur_workspace_size;
		void *cur_workspace;

		struct timespec start_time, end_time;
		clock_gettime(CLOCK_MONOTONIC, &start_time);

		if (to_offload[i])
			checkSemaphoreErrors(sem_wait(&sem_prefetch_done[i]));

		clock_gettime(CLOCK_MONOTONIC, &end_time);
		float lag = (end_time.tv_sec - start_time.tv_sec) * 1e3 + (end_time.tv_nsec - start_time.tv_nsec) * 1e-6;
		bwd_vdnn_lag.insert(bwd_vdnn_lag.begin(), lag);

		// {
		// 	int n;
		// 	std::cout << "waiting..\n";
		// 	std::cin >> n;
		// }

		if (i > 0) {
			if (layer_type[i] == ACTV or layer_type[i] == SOFTMAX) {
				dlayer_input[i] = dlayer_input[i + 1];
			}
			else {
				int layer_to_prefetch = findPrefetchLayer(i);
				if (layer_to_prefetch != -1) {
					checkCNMEM(cnmemMalloc(&layer_input[layer_to_prefetch], layer_input_size[layer_to_prefetch] * data_type_size, NULL));
					space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[layer_to_prefetch] * data_type_size);
					// std::cout << "Free bytes: " << free_bytes << std::endl;
					if (layer_to_prefetch != 0) {
						checkCudaErrors(cudaMemcpyAsync(layer_input[layer_to_prefetch], h_layer_input[layer_to_prefetch], 
														layer_input_size[layer_to_prefetch] * data_type_size, cudaMemcpyHostToDevice, stream_memory));
					}
					else {
						// std::cout << "transfer here\n";
						checkCudaErrors(cudaMemcpyAsync(layer_input[layer_to_prefetch], X, 
														layer_input_size[layer_to_prefetch] * data_type_size, cudaMemcpyHostToDevice, stream_memory));
						// std::cout << "transfer here\n";
					}
					checkCudaErrors(cudaEventRecord(event_prefetch_done[layer_to_prefetch], stream_memory));
					checkPThreadErrors(pthread_create(&thread_flag_prefetch_done[layer_to_prefetch], NULL, NeuralNet::threadFlagPrefetchDoneHelper, (void *)(&(layer_num[layer_to_prefetch]))));
					checkPThreadErrors(pthread_detach(thread_flag_prefetch_done[layer_to_prefetch]));
				}
				checkCNMEM(cnmemMalloc(&dlayer_input[i], layer_input_size[i] * data_type_size, NULL));
				space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[i] * data_type_size);
			}
			// std::cout << "Free bytes: " << free_bytes << std::endl;
		}
		// ---------------------- vDNN end ------------------------

		if (layer_type[i] == CONV) {
			// std::cout << "here\n";
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

			if (cur_params->activation_mode != ACTIVATION_NONE) {
				checkCUDNN(cudnnActivationBackward(cudnn_handle, cur_params->actv_desc, &alpha,
												cur_params->output_tensor, layer_input[i + 1],
												cur_params->output_tensor, dlayer_input[i + 1],
												cur_params->output_tensor, layer_input[i + 1],
												&beta,
												cur_params->output_tensor, dlayer_input[i + 1]));
			}

			// allocate space for derivative
			if (!pre_alloc_conv_derivative) {
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->kernel_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->C_out * data_type_size);
			}

			cur_filter_workspace_size = cur_params->bwd_filter_workspace_size;
			if (i > 0)
				cur_data_workspace_size = cur_params->bwd_data_workspace_size;
			else
				cur_data_workspace_size = 0;
			// std::cout << "bwd cur_workspace_size: " << cur_workspace_size << std::endl;
			cur_workspace_size = (cur_filter_workspace_size > cur_data_workspace_size) ? cur_filter_workspace_size : cur_data_workspace_size;
			checkCNMEM(cnmemMalloc(&cur_workspace, cur_workspace_size, NULL));

			checkCUDNN(cudnnConvolutionBackwardBias(cudnn_handle, &alpha,
													cur_params->output_tensor, dlayer_input[i + 1],
													&beta,
													cur_params->bias_desc, cur_params->db));

			// std::cout << "neural_net: backward conv i:" << i << std::endl;

			checkCUDNN(cudnnConvolutionBackwardFilter(cudnn_handle, &alpha,
														cur_params->input_tensor, layer_input[i],
														cur_params->output_tensor, dlayer_input[i + 1],
														cur_params->conv_desc, cur_params->bwd_filter_algo,
														cur_workspace, cur_workspace_size,
														&beta, 
														cur_params->filter_desc,
														cur_params->dW));
			if (i > 0)
				checkCUDNN(cudnnConvolutionBackwardData(cudnn_handle, &alpha,
														cur_params->filter_desc, cur_params->W,
														cur_params->output_tensor, dlayer_input[i + 1],
														cur_params->conv_desc, cur_params->bwd_data_algo,
														cur_workspace, cur_workspace_size,
														&beta,
														cur_params->input_tensor, dlayer_input[i]));

			space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
			// std::cout << "Free bytes: " << free_bytes << std::endl;
			// std::cout << "here\n";
			cur_params->stepParams(cublas_handle, learning_rate);
		}

		else if (layer_type[i] == FULLY_CONNECTED) {
			FCLayerParams *cur_params = (FCLayerParams *)params[i];

			if (cur_params->activation_mode != ACTIVATION_NONE) {
				checkCUDNN(cudnnActivationBackward(cudnn_handle, cur_params->actv_desc, &alpha,
												cur_params->output_tensor, layer_input[i + 1],
												cur_params->output_tensor, dlayer_input[i + 1],
												cur_params->output_tensor, layer_input[i + 1],
												&beta,
												cur_params->output_tensor, dlayer_input[i + 1]));
			}

			if (!pre_alloc_fc_derivative) {
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->weight_matrix_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->C_out * data_type_size);
			}

			if (data_type == CUDNN_DATA_FLOAT) {
				// bias backward
				checkCUBLAS(cublasSgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										cur_params->C_out, 1, batch_size,
										&Salpha,
										(float *)dlayer_input[i + 1], cur_params->C_out,
										(float *)one_vec, batch_size,
										&Sbeta,
										(float *)cur_params->db, cur_params->C_out));

				// weight backward
				checkCUBLAS(cublasSgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_T,
										cur_params->C_out, cur_params->C_in, batch_size,
										&Salpha,
										(float *)dlayer_input[i + 1], cur_params->C_out,
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
											(float *)dlayer_input[i + 1], cur_params->C_out,
											&Sbeta,
											(float *)dlayer_input[i], cur_params->C_in));
			}

			else if (data_type == CUDNN_DATA_DOUBLE) {
				// bias backward
				checkCUBLAS(cublasDgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										cur_params->C_out, 1, batch_size,
										&Dalpha,
										(double *)dlayer_input[i + 1], cur_params->C_out,
										(double *)one_vec, batch_size,
										&Dbeta,
										(double *)cur_params->db, cur_params->C_out));

				// weight backward
				checkCUBLAS(cublasDgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_T,
										cur_params->C_out, cur_params->C_in, batch_size,
										&Dalpha,
										(double *)dlayer_input[i + 1], cur_params->C_out,
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
											(double *)dlayer_input[i + 1], cur_params->C_out,
											&Dbeta,
											(double *)dlayer_input[i], cur_params->C_in));
			}
			cur_params->stepParams(cublas_handle, learning_rate);
		}

		else if (layer_type[i] == DROPOUT) {
			DropoutLayerParams *cur_params = (DropoutLayerParams *)params[i];
			checkCUDNN(cudnnDropoutBackward(cudnn_handle, cur_params->dropout_desc,
											cur_params->input_tensor, dlayer_input[i + 1],
											cur_params->input_tensor, dlayer_input[i],
											cur_params->reserved_space, cur_params->reserved_space_size));
		}

		else if (layer_type[i] == BATCHNORM) {
			BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];

			if (!pre_alloc_batch_norm_derivative) {
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->allocation_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->allocation_size * data_type_size);
			}

			checkCUDNN(cudnnBatchNormalizationBackward(cudnn_handle, cur_params->mode,
														&alpha, &beta,
														&alpha, &beta,
														cur_params->input_tensor, layer_input[i],
														cur_params->input_tensor, dlayer_input[i + 1],
														cur_params->input_tensor, dlayer_input[i],
														cur_params->sbmv_desc, cur_params->scale,
														cur_params->dscale, cur_params->dbias,
														cur_params->epsilon,
														cur_params->result_save_mean, cur_params->result_save_inv_var));

			cur_params->stepParams(cublas_handle, learning_rate);
		}

		else if (layer_type[i] == POOLING) {
			PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
			checkCUDNN(cudnnPoolingBackward(cudnn_handle, cur_params->pool_desc, &alpha, 
											cur_params->output_tensor, layer_input[i + 1],
											cur_params->output_tensor, dlayer_input[i + 1],
											cur_params->input_tensor, layer_input[i],
											&beta,
											cur_params->input_tensor, dlayer_input[i]));
		}

		else if (layer_type[i] == ACTV) {
			ActivationLayerParams *cur_params = (ActivationLayerParams *)params[i];
			checkCUDNN(cudnnActivationBackward(cudnn_handle, cur_params->actv_desc, &alpha,
												cur_params->input_tensor, layer_input[i + 1],
												cur_params->input_tensor, dlayer_input[i + 1],
												cur_params->input_tensor, layer_input[i],
												&beta,
												cur_params->input_tensor, dlayer_input[i]));
			continue;
		}

		else if (layer_type[i] == SOFTMAX) {
			// std::cout << "compute here\n";
			SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
			checkCUDNN(cudnnSoftmaxBackward(cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
											cur_params->input_tensor, layer_input[i + 1],
											cur_params->input_tensor, dlayer_input[i + 1],
											&beta,
											cur_params->input_tensor, dlayer_input[i]));
			// std::cout << "compute here\n";
			continue;
		}

		// ---------------------- vDNN start ----------------------
		
		// checkCudaErrors(cudaDeviceSynchronize());
		// struct timespec start_time, end_time;

		checkCudaErrors(cudaStreamSynchronize(stream_compute));

		// if (train)
		// 	clock_gettime(CLOCK_MONOTONIC, &start_time);

		// checkCudaErrors(cudaStreamSynchronize(stream_memory));
		// if (train) {
		// 	clock_gettime(CLOCK_MONOTONIC, &end_time);
		// 	float lag = (end_time.tv_sec - start_time.tv_sec) * 1e3 + (end_time.tv_nsec - start_time.tv_nsec) * 1e-6;
		// 	bwd_vdnn_lag.insert(bwd_vdnn_lag.begin(), lag);
		// }

		if (layer_type[i] == CONV) {
			checkCNMEM(cnmemFree(cur_workspace, NULL));
			space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
			if (!pre_alloc_conv_derivative) {
				ConvLayerParams *cur_params = (ConvLayerParams *)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->kernel_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->C_out * data_type_size);
			}
		}
		else if (layer_type[i] == FULLY_CONNECTED) {
			if (!pre_alloc_fc_derivative) {
				FCLayerParams *cur_params = (FCLayerParams *)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->weight_matrix_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->C_out * data_type_size);
			}
		}
		else if (layer_type[i] == BATCHNORM) {
			if (train == true and !pre_alloc_batch_norm_derivative) {
				BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->allocation_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->allocation_size * data_type_size);
			}
		}

		checkCNMEM(cnmemFree(layer_input[i + 1], NULL));
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i + 1] * data_type_size);
		checkCNMEM(cnmemFree(dlayer_input[i + 1], NULL));
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i + 1] * data_type_size);
		if (i == 0) {
			checkCNMEM(cnmemFree(layer_input[i], NULL));
			space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);
		}	
		// ---------------------- vDNN end ------------------------
	}
	if (space_tracker.getConsumed() != 0) {
		// std::cout << "Panic!! Space not updated properly\n";
	}

	// exit(0);
}