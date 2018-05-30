#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <helper_cuda.h>
#include <iostream>
#include <cmath>

#ifndef UTILS
#define UTILS

#define BW (16 * 16)
#define CNMEM_GRANULARITY 512

#define FatalError(s) do {                                             \
	std::stringstream _where, _message;                                \
	_where << __FILE__ << ':' << __LINE__;                             \
	_message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
	std::cerr << _message.str() << "\nAborting...\n";                  \
	cudaDeviceReset();                                                 \
	exit(1);                                                           \
} while(0)

#define checkCUDNN(expression)                               							\
{                                                            							\
	cudnnStatus_t status = (expression);                     							\
	if (status != CUDNN_STATUS_SUCCESS) {                    							\
	  std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "      \
				<< cudnnGetErrorString(status) << std::endl; 							\
	  std::exit(EXIT_FAILURE);                               							\
	}                                                        							\
}

#define checkCUBLAS(expression)                             							\
{                                                           							\
	cublasStatus_t status = (expression);                   							\
	if (status != CUBLAS_STATUS_SUCCESS) {                  							\
	  std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "     	\
				<< _cudaGetErrorEnum(status) << std::endl;  							\
	  std::exit(EXIT_FAILURE);                              							\
	}                                                       							\
}

#define checkCURAND(expression)                             							\
{                                                          								\
	curandStatus_t status = (expression);                   							\
	if (status != CURAND_STATUS_SUCCESS) {                  							\
	  std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "     	\
				<< _cudaGetErrorEnum(status) << std::endl;  							\
	  std::exit(EXIT_FAILURE);                              							\
	}                                                       							\
}

#define checkCNMEM(expression)                               							\
{                                                            							\
	cnmemStatus_t status = (expression);                     							\
	if (status != CNMEM_STATUS_SUCCESS) {                    							\
	  std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "      \
				<< cnmemGetErrorString(status) << std::endl; 							\
	  std::exit(EXIT_FAILURE);                               							\
	}                                                        							\
}

#define checkCNMEMRet(expression)                               						\
{                                                            							\
	cnmemStatus_t status = (expression);                     							\
	if (status != CNMEM_STATUS_SUCCESS) {                    							\
		if (status == CNMEM_STATUS_OUT_OF_MEMORY) {										\
			return false;																\
		}																				\
	  std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "      \
				<< cnmemGetErrorString(status) << std::endl; 							\
	  std::exit(EXIT_FAILURE);                               							\
	}                                                        							\
}

#define checkPThreadErrors(expression)													\
{																						\
	int status = (expression);															\
	if (status != 0) {																	\
		std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "	\
					<< "PthreadError" << std::endl;										\
		std::exit(EXIT_FAILURE);														\
	}																					\
}

#define checkSemaphoreErrors(expression)												\
{																						\
	int status = (expression);															\
	if (status != 0) {																	\
		std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "	\
					<< "SemaphoreError" << std::endl;									\
		std::exit(EXIT_FAILURE);														\
	}																					\
}

#define checkPMutexErrors(expression)													\
{																						\
	int status = (expression);															\
	if (status != 0) {																	\
		std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "	\
					<< "PMutexError" << std::endl;										\
		std::exit(EXIT_FAILURE);														\
	}																					\
}

#define checkPCondErrors(expression)													\
{																						\
	int status = (expression);															\
	if (status != 0) {																	\
		std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "	\
					<< "PCondError" << std::endl;										\
		std::exit(EXIT_FAILURE);														\
	}																					\
}

struct LayerDimension {
	int N, C, H, W;

	int getTotalSize();
};

template <typename T>
__global__ void fillValue(T *v, int size, int value) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;
	v[i] = value;
}

void outOfMemory();

struct CnmemSpace {
	long free_bytes;
	long initial_free_bytes;

	enum Op {ADD, SUB};

	CnmemSpace(long free_bytes);

	void updateSpace(Op op, long size);

	bool isAvailable();

	long getConsumed();

	void updateMaxConsume(size_t &max_consume);

};

#endif