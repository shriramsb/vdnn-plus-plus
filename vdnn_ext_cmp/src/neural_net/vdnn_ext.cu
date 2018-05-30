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

		// post a semaphore to indicate offload and compress are done
		checkSemaphoreErrors(sem_post(&sem_offload_done[layer_num]));
	}


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

		// call compression thread for this net object with layer_num
	}
}

void *NeuralNet::threadOffloadHandlerHelper(void *arg) {
	PtrIndex *ptr_index = static_cast<PtrIndex *>(arg);
	NeuralNet *net = static_cast<NeuralNet *>(ptr_index->ptr);
	int index = ptr_index->index;
	net->threadOffloadHandler(index);
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