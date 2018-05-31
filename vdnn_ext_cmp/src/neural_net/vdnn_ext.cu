#include "neural_net.h"
#include <time.h>

void NeuralNet::lockedcnmemMalloc(void **p, size_t size, cudaStream_t stream) {
	checkPMutexErrors(pthread_mutex_lock(&lock_cnmem_memory));
	while (true) {
		cnmemStatus_t status = cnmemMalloc(p, size, stream);
		if (status == CNMEM_STATUS_SUCCESS) {
			break;
		}
		else if (status == CNMEM_STATUS_OUT_OF_MEMORY) {
			// std::cout << "locked cnmem malloc, waiting for memory\n";
			checkPCondErrors(pthread_cond_wait(&cond_cnmem_available, &lock_cnmem_memory));
		}
	}
	checkPMutexErrors(pthread_mutex_unlock(&lock_cnmem_memory));
}

void NeuralNet::lockedcnmemFree(void *p, cudaStream_t stream) {
	checkPMutexErrors(pthread_mutex_lock(&lock_cnmem_memory));
	checkCNMEM(cnmemFree(p, stream));
	checkPCondErrors(pthread_cond_broadcast(&cond_cnmem_available));
	checkPMutexErrors(pthread_mutex_unlock(&lock_cnmem_memory));
}

// void NeuralNet::threadOffloadHandler(int layer_num) {
// 	checkCudaErrors(cudaEventSynchronize(event_offload_done[layer_num]));
// 	lockedcnmemFree(layer_input[layer_num], NULL);
// 	checkSemaphoreErrors(sem_post(&sem_sync_offload[layer_num]));
// 	// space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[layer_num] * data_type_size);
// }

// void *NeuralNet::threadOffloadHandlerHelper(void *arg) {
// 	PtrIndex *ptr_index = static_cast<PtrIndex *>(arg);
// 	NeuralNet *net = static_cast<NeuralNet *>(ptr_index->ptr);
// 	int index = ptr_index->index;
// 	net->threadOffloadHandler(index);
// 	return NULL;
// }

void NeuralNet::threadOffloadHandler(int layer_num) {
	
	if (layer_num > 0) {
		// allocate space in CPU memory
		checkCudaErrors(cudaMallocHost(h_layer_input[layer_num], layer_input_size[layer_num] * data_type_size));
		
		// start offload
		checkCudaErrors(cudaMemcpyAsync(h_layer_input[layer_num], layer_input[layer_num], 
												layer_input_size[layer_num] * data_type_size, cudaMemcpyDeviceToHost, stream_memory));
		// record event for finding point of offload completion
		checkCudaErrors(cudaEventRecord(event_offload_done[layer_num], stream_memory));

		// wait for offload to complete
		checkCudaErrors(cudaEventSynchronize(event_offload_done[layer_num]));
	}

	// post a semaphore to indicate offload and compress are done
	checkSemaphoreErrors(sem_post(&sem_offload_done[layer_num]));

	// if required,
	// push into compression_queue
	// broadcast on compress_job_available
	checkPMutexErrors(pthread_mutex_lock(&lock_compression_queue));
	if (to_compress[layer_num] && !fwd_compute_done) {
		compression_queue.push_back(layer_num);
		checkPCondErrors(pthread_cond_broadcast(cond_compression_job_available));
	}
	checkPMutexErrors(pthread_mutex_unlock(&lock_compression_queue));

	// wait for computation to complete
	checkCudaErrors(cudaEventSynchronize(event_fwd_compute_done[layer_num]));

	// free GPU memory
	lockedcnmemFree(layer_input[layer_num], NULL);



}

void *NeuralNet::threadOffloadHandlerHelper(void *arg) {
	PtrIndex *ptr_index = static_cast<PtrIndex *>(arg);
	NeuralNet *net = static_cast<NeuralNet *>(ptr_index->ptr);
	int index = ptr_index->index;
	net->threadOffloadHandler(index);
	return NULL;
}

void *NeuralNet::threadCompressionScheduler(void *arg) {
	NeuralNet *net = (NeuralNet *)arg;
	while (true) {
		checkPMutexErrors(pthread_mutex_lock(&net->lock_compression_queue));
		while (net->compression_queue.size() == 0) {
			checkPCondErrors(pthread_cond_wait(&net->compress_job_available, &net->lock_compression_queue));
		}
		int layer_num = net->compression_queue.front();
		net->compression_queue.pop();
		checkPMutexErrors(pthread_mutex_unlock(&net->lock_compression_queue));
		if (layer_num == -1)
			return NULL;
		// call compression thread for this net object with layer_num
		net->compress(layer_num);
	}
	return NULL;
}

void NeuralNet::compress(int layer_num) {
	// allocate space for mask
	checkCudaErrors(cudaMallocHost(compressed_data[layer_num].mask, ceil(1.0 * layer_input_size[layer_num] / COMPRESSION_BATCH_SIZE)));

	// start the threads
	pthread_t compression_threads[NUM_COMPRESSION_THREADS];
	for (int i = 0; i < NUM_COMPRESSION_THREADS; i++) {
		pthread_create(&compression_threads[i], NULL, compressThread, (void *)&compress_thread_args[layer_num][i]);
	}
	for (int i = 0; i < NUM_COMPRESSION_THREADS; i++) {
		pthread_join(compression_threads[i], NULL);
	}
	checkCudaErrors(cudaFreeHost(h_layer_input[layer_num]));
}

void *compressThread(void *arg) {
	CompressThreadArgs *args = (CompressThreadArgs *)arg;
	
	if (args->data_type == DATA_FLOAT) {
		// retrieve args
		float *original_data = (float *)(args->original_data);
		CompressedData compressed_data = *(args->compressed_data);
		CompressionMetadata compression_metadata = *(args->compression_metadata);
		int thread_num = args->thread_num;

		// reset compressed_data.slot_taken
		for (int i = 0; i < COMPRESSION_DISCRETIZATION_FACTOR; i++) {
			compressed_data.slot_taken[thread_num][i] = false;
		}

		Position2dArray compressed_data_pos, mask_pos;
		compressed_data_pos.slot = -1, compressed_data_pos.offset = -1;
		mask_pos.slot = compression_metadata.start_pos[thread_num] / COMPRESSION_BATCH_SIZE;
		mask_pos.offset = 0;

		for (long i = compression_metadata.start_pos[thread_num]; i < compression_metadata.start_pos[thread_num] + compression_metadata.num_elements[thread_num]; i++) {
			if (mask_pos.offset == 0)
				compressed_data.mask[mask_pos.slot] = 0;

			if (original_data[i] > 0) {
				if (compressed_data_pos.offset == -1 or compressed_data_pos.offset == compression_metadata.slot_size[thread_num][compressed_data_pos.slot]) {
					compressed_data_pos.slot += 1;
					checkCudaErrors(cudaMallocHost((void **)&compressed_data[thread_num][compressed_data_pos.slot], compression_metadata.slot_size[thread_num][compressed_data_pos.slot] * sizeof(float)));
					compressed_data.slot_taken[thread_num][compressed_data_pos.slot] = true;
					compressed_data_pos.offset = 0;
				}
				compressed_data.mask[mask_pos.slot] = (compressed_data.mask[mask_pos.slot] << 1) + 1;
				((float ***)compressed_data.data)[thread_num][compressed_data_pos.slot][compressed_data_pos.offset] = original_data[i];
				compressed_data_pos.offset += 1;
			}
			else {
				compressed_data.mask[mask_pos.slot] = (compressed_data.mask[mask_pos.slot] << 1);
			}
			mask_pos.offset += 1;
			if (mask_pos.offset == COMPRESSION_BATCH_SIZE) {
				mask_pos.slot += 1;
				mask_pos.offset = 0;
			}

		}
	}
	else {
		// retrieve args
		double *original_data = (double *)(args->original_data);
		CompressedData compressed_data = *(args->compressed_data);
		CompressionMetadata compression_metadata = *(args->compression_metadata);
		int thread_num = args->thread_num;

		// reset compressed_data.slot_taken
		for (int i = 0; i < COMPRESSION_DISCRETIZATION_FACTOR; i++) {
			compressed_data.slot_taken[thread_num][i] = false;
		}

		Position2dArray compressed_data_pos, mask_pos;
		compressed_data_pos.slot = -1, compressed_data_pos.offset = -1;
		mask_pos.slot = compression_metadata.start_pos[thread_num] / COMPRESSION_BATCH_SIZE;
		mask_pos.offset = 0;

		for (long i = compression_metadata.start_pos[thread_num]; i < compression_metadata.start_pos[thread_num] + compression_metadata.num_elements[thread_num]; i++) {
			if (mask_pos.offset == 0)
				compressed_data.mask[mask_pos.slot] = 0;

			if (original_data[i] > 0) {
				if (compressed_data_pos.offset == -1 or compressed_data_pos.offset == compression_metadata.slot_size[thread_num][compressed_data_pos.slot]) {
					compressed_data_pos.slot += 1;
					checkCudaErrors(cudaMallocHost((void **)&compressed_data[thread_num][compressed_data_pos.slot], compression_metadata.slot_size[thread_num][compressed_data_pos.slot] * sizeof(double)));
					compressed_data.slot_taken[thread_num][compressed_data_pos.slot] = true;
					compressed_data_pos.offset = 0;
				}
				compressed_data.mask[mask_pos.slot] = (compressed_data.mask[mask_pos.slot] << 1) + 1;
				((double ***)compressed_data.data)[thread_num][compressed_data_pos.slot][compressed_data_pos.offset] = original_data[i];
				compressed_data_pos.offset += 1;
			}
			else {
				compressed_data.mask[mask_pos.slot] = (compressed_data.mask[mask_pos.slot] << 1);
			}
			mask_pos.offset += 1;
			if (mask_pos.offset == COMPRESSION_BATCH_SIZE) {
				mask_pos.slot += 1;
				mask_pos.offset = 0;
			}

		}	
	}
	return NULL;
}

void NeuralNet::threadFlagPrefetchDone(int layer_num) {
	checkCudaErrors(cudaEventSynchronize(event_prefetch_done[layer_num]));
	checkSemaphoreErrors(sem_post(&sem_prefetch_done[layer_num]));
}

void *NeuralNet::threadFlagPrefetchDoneHelper(void *arg) {
	PtrIndex *ptr_index = static_cast<PtrIndex *>(arg);
	NeuralNet *net = static_cast<NeuralNet *>(ptr_index->ptr);
	int index = ptr_index->index;
	net->threadFlagPrefetchDone(index);
	return NULL;
}