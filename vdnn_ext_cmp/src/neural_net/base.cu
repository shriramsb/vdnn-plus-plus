#include "neural_net.h"
#include <time.h>

NeuralNet::NeuralNet(std::vector<LayerSpecifier> &layers, DataType data_type, int batch_size, TensorFormat tensor_format, 
						long long dropout_seed, float softmax_eps, float init_std_dev, vDNNType vdnn_type, vDNNConvAlgo vdnn_conv_algo, 
						UpdateRule update_rule) {
	
	// ---------------------- vDNN start ----------------------
	checkCudaErrors(cudaStreamCreate(&stream_compute));
	checkCudaErrors(cudaStreamCreate(&stream_memory));
	this->vdnn_type = vdnn_type;
	this->vdnn_conv_algo = vdnn_conv_algo;
	// ---------------------- vDNN end ------------------------

	// create handle
	checkCUDNN(cudnnCreate(&cudnn_handle));
	checkCUDNN(cudnnSetStream(cudnn_handle, stream_compute));

	checkCUBLAS(cublasCreate(&cublas_handle));
	checkCUBLAS(cublasSetStream(cublas_handle, stream_compute));

	checkCURAND(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
	checkCURAND(curandSetStream(curand_gen, stream_compute));

	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
	size_t init_free_bytes = free_bytes;
	std::cout << "Free bytes at start: " << free_bytes << std::endl;

	pre_alloc_conv_derivative = false;
	pre_alloc_fc_derivative = false;
	pre_alloc_batch_norm_derivative = true;

	if (vdnn_type == vDNN_NONE) {
		pre_alloc_conv_derivative = true;
		pre_alloc_fc_derivative = true;
		pre_alloc_batch_norm_derivative = true;
	}

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

	num_layers = layers.size();
	// allocation of space for input to each layer
	layer_input = (void **)malloc((num_layers + 1) * sizeof(void *));
	layer_input_size = (int *)malloc((num_layers + 1) * sizeof(int));
	dlayer_input = (void **)malloc((num_layers + 1) * sizeof(void *));
	params = (void **)malloc(num_layers * sizeof(void *));

	LayerDimension prev_output_size;
	LayerDimension current_output_size;
	for (int i = 0; i < num_layers; i++) {
		layer_type.push_back(layers[i].type);
		if (layers[i].type == CONV) {
			ConvDescriptor *user_params = (ConvDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(ConvLayerParams));
			((ConvLayerParams *)params[i])->initializeValues(cudnn_handle, user_params, this->data_type, batch_size, this->tensor_format, 
																data_type_size, current_output_size, update_rule);
		}
		else if (layers[i].type == FULLY_CONNECTED) {
			FCDescriptor *user_params = (FCDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(FCLayerParams));
			((FCLayerParams *)params[i])->initializeValues(user_params, batch_size, this->tensor_format, this->data_type, 
															current_output_size, update_rule);
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
			((BatchNormLayerParams *)params[i])->initializeValues(user_params, this->data_type, this->tensor_format, batch_size, 
																	current_output_size, update_rule);
			
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

	// ---------------------- vDNN start ----------------------

	// allocate space in host memory for layers to be transferred
	h_layer_input = (void **)malloc(num_layers * sizeof(void *));
	to_offload = (bool *)malloc(num_layers * sizeof(bool));
	prefetched = (bool *)malloc(num_layers * sizeof(bool));

	// ---------------------- vDNN end ------------------------
	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
	std::cout << "Free bytes just before allocate space: " << free_bytes << std::endl;
	// allocate space for parameters
	// Exception BatchNorm - looks like it will take lots of space if only FC layers - space taken = size of one input
	for (int i = 0; i < num_layers; i++) {
		size_t input_size;
		if (layers[i].type == CONV) {
			ConvDescriptor *user_params = (ConvDescriptor *)layers[i].params;
			((ConvLayerParams *)params[i])->allocateSpace(curand_gen, this->data_type, data_type_size, init_std_dev, 
															free_bytes, pre_alloc_conv_derivative);

			input_size = batch_size * user_params->input_channels * user_params->input_h * user_params->input_w;
			if (i == 0) {
				input_channels = user_params->input_channels;
				input_h = user_params->input_h;
				input_w = user_params->input_w;
			}
		}
		else if (layers[i].type == FULLY_CONNECTED) {
			FCDescriptor *user_params = (FCDescriptor *)layers[i].params;
			((FCLayerParams *)params[i])->allocateSpace(curand_gen, this->data_type, data_type_size, init_std_dev, 
														free_bytes, pre_alloc_fc_derivative);
			input_size = batch_size * user_params->input_channels;
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
			if (i == 0) {
				input_channels = user_params->channels;
				input_h = user_params->h;
				input_w = user_params->w;
			}
		}
		else if (layers[i].type == BATCHNORM) {
			BatchNormDescriptor *user_params = (BatchNormDescriptor *)layers[i].params;
			((BatchNormLayerParams *)params[i])->allocateSpace(this->data_type, data_type_size, 
																free_bytes, pre_alloc_batch_norm_derivative);
			input_size = batch_size * user_params->channels * user_params->h * user_params->w;
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

			// assuming this is last layer, allocate for next layer as well
			// checkCudaErrors(cudaMalloc(&layer_input[i + 1], input_size * data_type_size));
			// checkCudaErrors(cudaMalloc(&dlayer_input[i + 1], input_size * data_type_size));
			layer_input_size[i + 1] = input_size;
			if (i == 0) {
				input_channels = user_params->channels;
				input_h = user_params->h;
				input_w = user_params->w;
			}
			if (i == num_layers - 1) {
				num_classes = user_params->channels;
			}
		}

		// do not allocate memory initially
		// checkCudaErrors(cudaMalloc(&layer_input[i], input_size * data_type_size));
		// checkCudaErrors(cudaMalloc(&dlayer_input[i], input_size * data_type_size));
		
		// ---------------------- vDNN start ----------------------
		layer_input_size[i] = input_size;
		// ---------------------- vDNN end ------------------------
	}
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
	std::cout << "Free bytes just after allocate space: " << free_bytes << std::endl;
	// very small - could be allocated initially itself
	checkCudaErrors(cudaMalloc((void **)&y, batch_size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&pred_y, batch_size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&loss, batch_size * sizeof(float)));
	checkCudaErrors(cudaMalloc(&one_vec, batch_size * data_type_size));

	if (this->data_type == CUDNN_DATA_FLOAT)
		fillValue<float><<<ceil(1.0 * batch_size / BW), BW>>>((float *)one_vec, batch_size, 1);
	else
		fillValue<double><<<ceil(1.0 * batch_size / BW), BW>>>((double *)one_vec, batch_size, 1);
	
	checkCudaErrors(cudaMallocHost((void **)&h_loss, batch_size * sizeof(float)));
	checkCudaErrors(cudaMallocHost((void **)&h_pred_y, batch_size * sizeof(int)));
	
	// do not allocate workspace initially
	// allocate space for workspace and also keep track of algo
	// size_t cur_workspace_size;
	// workspace_size = 0;
	// for (int i = 0; i < num_layers; i++) {
	// 	if (layers[i].type == CONV) {
	// 		((ConvLayerParams *)params[i])->getWorkspaceSize(cur_workspace_size, free_bytes);
	// 		if (cur_workspace_size > workspace_size)
	// 			workspace_size = cur_workspace_size;
	// 	}
	// }

	// checkCudaErrors(cudaMalloc(&workspace, workspace_size));
	// free_bytes = free_bytes - workspace_size;
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));

	// leave 600 MB and use the rest
	std::cout << "Free bytes: " << free_bytes << std::endl;
	free_bytes -= 1024 * 1024 * 600;
	// ---------------------- vDNN start ----------------------
	size_t exp_max_consume, max_consume;
	vDNNOptimize(exp_max_consume, max_consume);
	std::cout << "actual_max_consume: " << max_consume << std::endl;
	std::cout << "exp_max_consume: " << exp_max_consume << std::endl;
	std::cout << "diff_max_consume(MB): " << (max_consume - exp_max_consume) / (1.0 * 1024 * 1024) << std::endl;
	std::cout << "exp_free_bytes(MB): " << (free_bytes + 1024 * 1024 * 600 - exp_max_consume) / (1.0 * 1024 * 1024) << std::endl;
	std::cout << "exp_total_consume(MB): " << (init_free_bytes - (free_bytes + 600 * 1024 * 1024 - exp_max_consume)) / (1.0 * 1024 * 1024) << std::endl;
	std::cout << "actual_total_consume(MB): " << (init_free_bytes - (free_bytes + 600 * 1024 * 1024 - max_consume)) / (1.0 * 1024 * 1024) << std::endl;

	// ---------------------- vDNN end ------------------------


	// ---------------------- vDNN start ----------------------

	free_bytes = max_consume;

	cnmemDevice_t cnmem_device;
	size_t cnmem_stream_memory_size = free_bytes;

	cnmem_device.device = 0;
	cnmem_device.size = cnmem_stream_memory_size;
	cnmem_device.numStreams = 0;
	cnmem_device.streams = NULL;
	cnmem_device.streamSizes = NULL;

	// do not allow call to cudaMalloc
	checkCNMEM(cnmemInit(1, &cnmem_device, CNMEM_FLAGS_CANNOT_GROW));
	// ---------------------- vDNN end ------------------------

	// ---------------------- vDNN start ----------------------
	for (int i = 0; i < num_layers; i++) {
		std::cerr << "to_offload[i] " << to_offload[i] << std::endl;
	}

	for (int i = 0; i < num_layers; i++) {
		// allocate pinned memory in host
		if (to_offload[i])
			checkCudaErrors(cudaMallocHost(&h_layer_input[i], layer_input_size[i] * data_type_size));
	}
	// ---------------------- vDNN end ------------------------
	checkCudaErrors(cudaDeviceSynchronize());
	size_t temp_free_bytes;
	checkCudaErrors(cudaMemGetInfo(&temp_free_bytes, &total_bytes));
	std::cout << "Free bytes just before end of NeuralNet: " << temp_free_bytes << std::endl;
	// {
	// 	int n;
	// 	std::cout << "waiting..\n";
	// 	std::cin >> n;
	// }

	// ---------------------- vDNN ext start ------------------------
	event_offload_done = (cudaEvent_t *)malloc(num_layers * sizeof(cudaEvent_t));
	// thread_free_layer_input = (pthread_t *)malloc(num_layers * sizeof(pthread_t));

	event_prefetch_done = (cudaEvent_t *)malloc(num_layers * sizeof(cudaEvent_t));
	thread_flag_prefetch_done = (pthread_t *)malloc(num_layers * sizeof(pthread_t));
	sem_prefetch_done = (sem_t *)malloc(num_layers * sizeof(sem_t));

	for (int i = 0; i < num_layers; i++) {
		if (to_offload[i]) {
			checkCudaErrors(cudaEventCreate(&event_offload_done[i]));
			checkSemaphoreErrors(sem_init(&sem_offload_done[i], 0, 0));

			checkCudaErrors(cudaEventCreate(&event_prefetch_done[i]));
			checkSemaphoreErrors(sem_init(&sem_prefetch_done[i], 0, 0));
		}
	}
	checkPMutexErrors(pthread_mutex_init(&lock_cnmem_memory, NULL));
	checkPCondErrors(pthread_cond_init(&cond_cnmem_available, NULL));

	layer_num = (PtrIndex *)malloc(num_layers * sizeof(PtrIndex));
	for (int i = 0; i < num_layers; i++) {
		layer_num[i].ptr = this;
		layer_num[i].index = i;
	}

	// ---------------------- vDNN ext end ------------------------

	// ---------------------- vDNN ext cmp start ------------------

	to_compress = (bool *)malloc(num_layers * sizeof(bool));
	thread_offload_handler = (pthread_t *)malloc(num_layers * sizeof(pthread_t));
	sem_offload_done = (sem_t *)malloc(num_layers * sizeof(sem_t));
	event_fwd_compute_done = (cudaEvent_t *)malloc(num_layers * sizeof(cudaEvent_t));
	// init to_compress - choose layers to compress

	fwd_compute_done = false;
	
	sem_compress_done = (sem_t *)malloc(num_layers * sizeof(sem_t));

	for (int i = 0; i < num_layers; i++) {
		checkCudaErrors(cudaEventCreate(&event_fwd_compute_done[i]));
		if (to_compress[i]) {
			checkSemaphoreErrors(sem_init(&sem_compress_done[i]));
		}
	}

	checkPThreadErrors(pthread_create(&thread_compression_scheduler, NULL, NeuralNet::threadCompressionScheduler, (void *)(this)));
	checkPThreadErrors(pthread_detach(thread_compression_scheduler));

	checkPMutexErrors(pthread_mutex_init(&lock_compression_queue, NULL));
	checkPCondErrors(pthread_cond_init(&cond_compression_job_available, NULL));

	

	// ---------------------- vDNN ext cmp end --------------------

}

bool NeuralNet::simulateNeuralNetworkMemory(vDNNConvAlgoPref algo_pref, bool hard, size_t &exp_max_consume, size_t &max_consume) {
	CnmemSpace space_tracker(free_bytes);
	max_consume = 0;
	// forward pass
	// allocate space for 1st input
	std::cerr << "Initial Used space(MB): " << space_tracker.getConsumed() << std::endl;
	space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[0] * data_type_size);
	space_tracker.updateMaxConsume(max_consume);
	std::cerr << "Used space after allocating input(MB): " << space_tracker.getConsumed() << std::endl;
	
	std::cerr << "Forward pass" << std::endl;
	for (int i = 0; i < num_layers; i++) {
		if (layer_type[i] == SOFTMAX)
			break;

		std::cerr << "Processing layer " << i << std::endl;

		std::cerr << "Initial Used space(MB): " << space_tracker.getConsumed() << std::endl;
		space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[i + 1] * data_type_size);
		std::cerr << "Used space after output allocation(MB): " << space_tracker.getConsumed() << std::endl;
		space_tracker.updateMaxConsume(max_consume);

		if (layer_type[i] == CONV) {
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

			long cur_workspace_size = cur_params->getWorkspaceSize(space_tracker.free_bytes, ConvLayerParams::FWD, algo_pref, hard);
			space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
			space_tracker.updateMaxConsume(max_consume);

			if (cur_workspace_size == -1 or !space_tracker.isAvailable())
				return false;
			std::cerr << "Used space after workspace allocation(MB): " << space_tracker.getConsumed() << std::endl;

			// current layer computation over, deallocate workspace
			space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
			std::cerr << "Used space after workspace deallocation(MB): " << space_tracker.getConsumed() << std::endl;
		}



		if (!space_tracker.isAvailable())
			return false;
		// deallocate layer input
		if (to_offload[i]) {
			std::cerr << "deallocating input to " << i << std::endl;
			space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);
			std::cerr << "Used space after deallocating input(MB): " << space_tracker.getConsumed() << std::endl;
		}
	}

	std::cerr << "Backward pass" << std::endl;
	if (batch_size * num_classes * data_type_size != layer_input_size[num_layers] * data_type_size) {
		std::cout << "Panic!! Using wrong size\n";
		exit(0);
	}	
	// backward pass
	space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[num_layers] * data_type_size);
	std::cerr << "Used space after allocating final derivative(MB): " << space_tracker.getConsumed() << std::endl;
	space_tracker.updateMaxConsume(max_consume);
	// std::cerr << "max_consume: " << max_consume << std::endl;
	for (int i = num_layers - 1; i >= 0; i--) {
		// allocate space for previous layer derivative
		std::cerr << "Processing layer " << i << std::endl;
		std::cerr << "Used space initial(MB): " << space_tracker.getConsumed() << std::endl;
		if (i > 0) {
			if (layer_type[i] == SOFTMAX)
				continue;
			else {
				space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[i] * data_type_size);
				std::cerr << "Used space after allocating prev. derivative(MB): " << space_tracker.getConsumed() << std::endl;
				space_tracker.updateMaxConsume(max_consume);
			}
			// std::cerr << "max_consume: " << max_consume << std::endl;
		}

		int layer_to_prefetch = findPrefetchLayer(i);
		// if layer to be prefetched, allocate space for that layer
		if (layer_to_prefetch != -1) {
			std::cerr << "Prefetch layer " << layer_to_prefetch << std::endl;
			space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[layer_to_prefetch] * data_type_size);
			std::cerr << "Used space after allocating prefetch(MB): " << space_tracker.getConsumed() << std::endl;
			space_tracker.updateMaxConsume(max_consume);
		}

		if (layer_type[i] == CONV) {
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];
			long cur_filter_workspace_size = cur_params->getWorkspaceSize(space_tracker.free_bytes, ConvLayerParams::BWD_FILTER, algo_pref, hard);
			long cur_data_workspace_size = 0;
			if (i > 0)
				cur_data_workspace_size = cur_params->getWorkspaceSize(space_tracker.free_bytes, ConvLayerParams::BWD_DATA, algo_pref, hard);

			long cur_workspace_size = (cur_filter_workspace_size > cur_data_workspace_size) ? cur_filter_workspace_size :cur_data_workspace_size;

			space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
			std::cerr << "Used space after allocating workspace(MB): " << space_tracker.getConsumed() << std::endl;
			space_tracker.updateMaxConsume(max_consume);
			
			if (!pre_alloc_conv_derivative) {
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->kernel_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->C_out * data_type_size);
				space_tracker.updateMaxConsume(max_consume);
				std::cerr << "Used space after allocating weight derv.(MB): " << space_tracker.getConsumed() << std::endl;
			}

			// std::cerr << "max_consume: " << max_consume << std::endl;
			if (cur_filter_workspace_size == -1 or cur_data_workspace_size == -1 or !space_tracker.isAvailable())
				return false;

			// current layer computation over, deallocate workspace
			space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
			std::cerr << "Used space after deallocating workspace(MB): " << space_tracker.getConsumed() << std::endl;

			if (!pre_alloc_conv_derivative) {
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->kernel_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->C_out * data_type_size);
				std::cerr << "Used space after deallocating weight derv.(MB): " << space_tracker.getConsumed() << std::endl;
			}
		}
		else if (layer_type[i] == FULLY_CONNECTED) {
			FCLayerParams *cur_params = (FCLayerParams *)params[i];

			if (!pre_alloc_fc_derivative) {
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->weight_matrix_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->C_out * data_type_size);
				space_tracker.updateMaxConsume(max_consume);
				std::cerr << "Used space after allocating weight derv.(MB): " << space_tracker.getConsumed() << std::endl;
			}

			if (!space_tracker.isAvailable())
				return false;

			if (!pre_alloc_fc_derivative) {
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->weight_matrix_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->C_out * data_type_size);
				std::cerr << "Used space after deallocating weight derv.(MB): " << space_tracker.getConsumed() << std::endl;
			}
		}
		else if (layer_type[i] == BATCHNORM) {
			BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];
			if (!pre_alloc_batch_norm_derivative) {
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->allocation_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->allocation_size * data_type_size);
				space_tracker.updateMaxConsume(max_consume);
				std::cerr << "Used space after allocating weight derv.(MB): " << space_tracker.getConsumed() << std::endl;
			}

			if (!space_tracker.isAvailable())
				return false;

			if (!pre_alloc_batch_norm_derivative) {
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->allocation_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->allocation_size * data_type_size);
				std::cerr << "Used space after deallocating weight derv.(MB): " << space_tracker.getConsumed() << std::endl;
			}
		}

		if (!space_tracker.isAvailable())
			return false;
		// deallocate layer output and derivative
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i + 1] * data_type_size);
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i + 1] * data_type_size);
		std::cerr << "Used space after deallocating output, derivative(MB): " << space_tracker.getConsumed() << std::endl;
		// if 1st layer, deallocate input layer also
		if (i == 0) {
			space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);
			std::cerr << "Used space after deallocating input(MB): " << space_tracker.getConsumed() << std::endl;
		}
	}
	if (space_tracker.getConsumed() > 0)
		std::cerr << "Panic!! more free bytes\n";
	if (space_tracker.getConsumed() != 0)
		std::cerr << "Panic!! bytes not freed properly\n";
	// return true;

	exp_max_consume = max_consume;
	cnmemDevice_t cnmem_device;

	cnmem_device.device = 0;
	cnmem_device.size = max_consume;
	cnmem_device.numStreams = 0;
	cnmem_device.streams = NULL;
	cnmem_device.streamSizes = NULL;

	checkCNMEM(cnmemInit(1, &cnmem_device, 0));
	// check with cnmem once
	bool ret_val = simulateCNMEMMemory(max_consume);
	checkCNMEM(cnmemFinalize());
	return ret_val;
}

bool NeuralNet::simulateCNMEMMemory(size_t &max_consume) {

	resetPrefetched();

	checkCNMEMRet(cnmemMalloc(&layer_input[0], layer_input_size[0] * data_type_size, NULL));

	// forward propagate
	for (int i = 0; i < num_layers; i++) {
		size_t cur_workspace_size;
		void *cur_workspace;

		checkCNMEMRet(cnmemMalloc(&layer_input[i + 1], layer_input_size[i + 1] * data_type_size, NULL));
		if (layer_type[i] == CONV) {
			// std::cout << "conv\n";
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

			cur_workspace_size = cur_params->fwd_workspace_size;
			checkCNMEMRet(cnmemMalloc(&cur_workspace, cur_workspace_size, NULL));			
		}		
		
		if (layer_type[i] == CONV) {
			checkCNMEMRet(cnmemFree(cur_workspace, NULL));
		}

		if (to_offload[i]) {
			checkCNMEMRet(cnmemFree(layer_input[i], NULL));
		}

		if (layer_type[i + 1] == ACTV or layer_type[i + 1] == SOFTMAX) {
			i = i + 1;
		}
	}

	checkCNMEMRet(cnmemMalloc(&dlayer_input[num_layers], batch_size * num_classes * data_type_size, NULL));

	for (int i = num_layers - 1; i >= 0; i--) {
		// ---------------------- vDNN start ----------------------
		int cur_filter_workspace_size, cur_data_workspace_size, cur_workspace_size;
		void *cur_workspace;

		if (i > 0) {
			if (layer_type[i] == ACTV or layer_type[i] == SOFTMAX) {
				dlayer_input[i] = dlayer_input[i + 1];
			}
			else {
				int layer_to_prefetch = findPrefetchLayer(i);
				if (layer_to_prefetch != -1) {
					checkCNMEMRet(cnmemMalloc(&layer_input[layer_to_prefetch], layer_input_size[layer_to_prefetch] * data_type_size, NULL));
				}
				checkCNMEMRet(cnmemMalloc(&dlayer_input[i], layer_input_size[i] * data_type_size, NULL));
			}
		}

		if (layer_type[i] == CONV) {
			// std::cout << "here\n";
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

			// allocate space for derivative
			if (!pre_alloc_conv_derivative) {
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
			}

			cur_filter_workspace_size = cur_params->bwd_filter_workspace_size;
			if (i > 0)
				cur_data_workspace_size = cur_params->bwd_data_workspace_size;
			else
				cur_data_workspace_size = 0;
			cur_workspace_size = (cur_filter_workspace_size > cur_data_workspace_size) ? cur_filter_workspace_size : cur_data_workspace_size;
			checkCNMEMRet(cnmemMalloc(&cur_workspace, cur_workspace_size, NULL));
		}

		else if (layer_type[i] == FULLY_CONNECTED) {
			FCLayerParams *cur_params = (FCLayerParams *)params[i];

			if (!pre_alloc_fc_derivative) {
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
			}
		}

		else if (layer_type[i] == BATCHNORM) {
			BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];

			if (!pre_alloc_batch_norm_derivative) {
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
			}
		}

		else if (layer_type[i] == SOFTMAX) {
			// std::cout << "compute here\n";
			SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
			continue;
		}

		if (layer_type[i] == CONV) {
			checkCNMEMRet(cnmemFree(cur_workspace, NULL));
			if (!pre_alloc_conv_derivative) {
				ConvLayerParams *cur_params = (ConvLayerParams *)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
			}
		}
		else if (layer_type[i] == FULLY_CONNECTED) {
			if (!pre_alloc_fc_derivative) {
				FCLayerParams *cur_params = (FCLayerParams *)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
			}
		}
		else if (layer_type[i] == BATCHNORM) {
			if (!pre_alloc_batch_norm_derivative) {
				BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
			}
		}

		checkCNMEMRet(cnmemFree(layer_input[i + 1], NULL));
		checkCNMEMRet(cnmemFree(dlayer_input[i + 1], NULL));
		if (i == 0) {
			checkCNMEMRet(cnmemFree(layer_input[i], NULL));
		}	
	}
	size_t temp;
	checkCNMEM(cnmemMemGetInfo(&temp, &max_consume, NULL));
	return true;
}

void NeuralNet::vDNNOptimize(size_t &exp_max_consume, size_t &max_consume) {

	bool hard = true, soft = false;

	// if type is vDNN_ALL or vDNN_CONV, check if sufficient space is available
	if (vdnn_type == vDNN_ALL) {
		setOffload(OFFLOAD_ALL);
		resetPrefetched();
		if (vdnn_conv_algo == vDNN_PERFORMANCE_OPTIMAL) {
			if (!simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume))
				outOfMemory();
		}
		else if (vdnn_conv_algo == vDNN_MEMORY_OPTIMAL) {
			if (!simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume))
				outOfMemory();
		}

		return;
	}
	else if (vdnn_type == vDNN_CONV) {
		setOffload(OFFLOAD_CONV);
		resetPrefetched();
		if (vdnn_conv_algo == vDNN_PERFORMANCE_OPTIMAL) {
			if (!simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume))
				outOfMemory();
		}
		else if (vdnn_conv_algo == vDNN_MEMORY_OPTIMAL) {
			if (!simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume))
				outOfMemory();
		}

		return;
	}
	else if (vdnn_type == vDNN_NONE) {
		setOffload(OFFLOAD_NONE);
		resetPrefetched();
		if (vdnn_conv_algo == vDNN_PERFORMANCE_OPTIMAL) {
			if (!simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume))
				outOfMemory();
		}
		else if (vdnn_conv_algo == vDNN_MEMORY_OPTIMAL) {
			if (!simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume))
				outOfMemory();
		}

		return;	
	}

	if (vdnn_type == vDNN_DYN) {
	
		// check for trainability
		std::cerr << "vDNN_DYN\n";
		setOffload(NeuralNet::OFFLOAD_ALL);
		resetPrefetched();
		if(!simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume))
			outOfMemory();
	
		// check if work with fastest algo and no offload, if so, select it and return
		setOffload(NeuralNet::OFFLOAD_NONE);
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume)) {
			std::cerr << "Choosing PERF_OPT, NO OFFLOAD\n";
			return;
		}
	
		// check if conv offload and fastest algo works, then check if all offload and fastest algo works
		setOffload(NeuralNet::OFFLOAD_CONV);
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume)) {
			std::cerr << "Choosing PERF_OPT, CONV OFFLOAD\n";
			return;
		}
	
		setOffload(NeuralNet::OFFLOAD_ALL);
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume)) {
			std::cerr << "Choosing PERF_OPT, ALL OFFLOAD\n";
			return;
		}
	
		// optimize using greedy algo memory usage while improving performance
		setOffload(NeuralNet::OFFLOAD_CONV);
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, soft, exp_max_consume, max_consume)) {
			std::cerr << "Choosing GREEDY, CONV OFFLOAD\n";
			return;
		}
	
		setOffload(NeuralNet::OFFLOAD_ALL);
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, soft, exp_max_consume, max_consume)) {
			std::cerr << "Choosing GREEDY, ALL OFFLOAD\n";
			return;
		}

		setOffload(NeuralNet::OFFLOAD_CONV);
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume)) {
			std::cerr << "Choosing MEM_OPT, CONV OFFLOAD\n";
			return;
		}
	
		
		setOffload(NeuralNet::OFFLOAD_ALL);
		resetPrefetched();
		if(simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume)) {
			std::cerr << "Choosing MEM_OPT, ALL OFFLOAD\n";
			return;
		}
	}
	exit(0);

}

void NeuralNet::setOffload(NeuralNet::OffloadType offload_type) {
	if (offload_type == OFFLOAD_NONE) {
		for (int i = 0; i < num_layers; i++)
			to_offload[i] = false;
	}
	else if (offload_type == OFFLOAD_CONV) {
		for (int i = 0; i < num_layers; i++) {
			if (layer_type[i] == CONV)
				to_offload[i] = true;
			else
				to_offload[i] = false;
		}
		// set last non SOFTMAX/ACTV layer to no_offload
		for (int i = num_layers - 1; i >= 0; i--) {
			if (layer_type[i] == SOFTMAX or layer_type[i] == ACTV)
				;
			else {
				to_offload[i] = false;
				break;
			}
		}
	}
	else if (offload_type == OFFLOAD_ALL) {
		for (int i = 0; i < num_layers; i++) {
			if (layer_type[i] == ACTV or layer_type[i] == SOFTMAX)
				to_offload[i] = false;
			else
				to_offload[i] = true;
		}
		// set last non SOFTMAX/ACTV layer to no_offload
		for (int i = num_layers - 1; i >= 0; i--) {
			if (layer_type[i] == SOFTMAX or layer_type[i] == ACTV)
				;
			else {
				to_offload[i] = false;
				break;
			}
		}
	}
}

void NeuralNet::resetPrefetched() {
	for (int i = 0; i < num_layers; i++)
		prefetched[i] = false;
}

int NeuralNet::findPrefetchLayer(int cur_layer) {
	for (int i = cur_layer - 1; i >= 0; i--) {
		if (to_offload[i] && !prefetched[i]) {
			prefetched[i] = true;
			return i;
		}
		else if (layer_type[i] == CONV) {
			return -1;
		}
	}
	return -1;
}