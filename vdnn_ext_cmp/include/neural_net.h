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

// ---------------------- vDNN start ----------------------
#include <cnmem.h>
// ---------------------- vDNN emd ------------------------

// ---------------------- vDNN ext start ----------------------
#include <pthread.h>
#include <semaphore.h>
// ---------------------- vDNN ext end ------------------------

// ---------------------- vDNN ext cmp start ------------------
#include <queue>
#include <list>
#define NUM_COMPRESSION_THREADS 8
#define COMPRESSION_DISCRETIZATION_FACTOR 8
#define COMPRESSION_BATCH_SIZE 32

// ---------------------- vDNN ext cmp end --------------------

#ifndef NEURAL_NET
#define NEURAL_NET

struct PtrIndex {
	void *ptr;
	int index;
};

struct CompressedData {
	void ***data;
	bool **slot_taken;
	unsigned int *mask;
};

struct CompressionMetadata {
	long total_compression_batches;
	long *num_elements, *start_pos;
	long **slot_size; 
};

struct CompressionThreadArgs {
	CompressedData *compressed_data;
	void *original_data;
	CompressionMetadata *compression_metadata;
	int thread_num;
	DataType data_type;

};

struct Position2dArray {
	long slot;
	long offset;
};

class NeuralNet {
public:
	void **layer_input, **dlayer_input, **params;
	int *layer_input_size;
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
	
	// vDNN
	vDNNType vdnn_type;
	vDNNConvAlgo vdnn_conv_algo;
	cudaStream_t stream_compute, stream_memory;

	bool pre_alloc_conv_derivative, pre_alloc_fc_derivative, pre_alloc_batch_norm_derivative;

	void **h_layer_input;
	bool *to_offload, *prefetched;

	// ---------------------- vDNN ext start ------------------------
	cudaEvent_t *event_offload_done;

	pthread_mutex_t lock_cnmem_memory;
	pthread_cond_t cond_cnmem_available;

	// pass as argument to thread function
	PtrIndex *layer_num;

	void lockedcnmemMalloc(void **p, size_t size, cudaStream_t stream);
	void lockedcnmemFree(void *p, cudaStream_t stream);

	cudaEvent_t *event_prefetch_done;
	pthread_t *thread_flag_prefetch_done;

	sem_t *sem_prefetch_done;

	static void *threadFlagPrefetchDoneHelper(void *arg);
	void threadFlagPrefetchDone(int index);
	CompressedData *compressed_data;

	CompressionThreadArgs **compression_thread_args;

	// ---------------------- vDNN ext end ------------------------

	// ---------------------- vDNN ext cmp start ------------------

	bool *to_compress;
	
	pthread_t *thread_offload_handler;
	sem_t *sem_offload_done;

	static void *threadOffloadHandlerHelper(void *arg);
	void threadOffloadHandler(int layer_num);


	cudaEvent_t *event_fwd_compute_done;
	bool fwd_compute_done;
	
	pthread_t thread_compression_scheduler;
	static void *threadCompressionScheduler(void *arg);
	sem_t *sem_compress_done;

	std::queue<int, std::list<int> > compression_queue;
	pthread_mutex_t lock_compression_queue;
	pthread_cond_t cond_compression_job_available;

	CompressedData *compressed_data;
	CompressionMetadata *compression_metadata;

	// ---------------------- vDNN ext cmp end --------------------

	enum OffloadType {OFFLOAD_ALL, OFFLOAD_NONE, OFFLOAD_CONV};

	NeuralNet(std::vector<LayerSpecifier> &layers, DataType data_type, int batch_size, TensorFormat tensor_format, 
				long long dropout_seed, float softmax_eps, float init_std_dev, vDNNType vdnn_type, vDNNConvAlgo vdnn_conv_algo, 
				UpdateRule update_rule);

	void getLoss(void *X, int *y, double learning_rate, std::vector<float> &fwd_vdnn_lag, std::vector<float> &bwd_vdnn_lag, bool train = true, int *correct_count = NULL, float *loss = NULL);
	void getLoss(void *X, int *y, double learning_rate, bool train = true, int *correct_count = NULL, float *loss = NULL);

	void compareOutputCorrect(int *correct_count, int *y);

	float computeLoss();

	int findPrefetchLayer(int cur_layer);

	bool simulateNeuralNetworkMemory(vDNNConvAlgoPref algo_pref, bool hard, size_t &exp_max_consume, size_t &max_consume);
	bool simulateCNMEMMemory(size_t &max_consume);
	void vDNNOptimize(size_t &exp_max_consume, size_t &max_consume);
	void setOffload(OffloadType offload_type);
	void resetPrefetched();
};

#endif