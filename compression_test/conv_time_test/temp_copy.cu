#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>

// #include <opencv2/opencv.hpp>
#include <helper_cuda.h>

typedef unsigned char uchar;

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)


// #define checkCudaErrors(status) do {                                   \
//     std::stringstream _error;                                          \
//     if (status != 0) {                                                 \
//       _error << "Cuda failure: " << status;                            \
//       FatalError(_error.str());                                        \
//     }                                                                  \
// } while(0)

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#define checkCUBLAS(expression)                              \
  {                                                          \
    cublasStatus_t status = (expression);                    \
    if (status != CUBLAS_STATUS_SUCCESS) {                   \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << _cudaGetErrorEnum(status) << std::endl;   \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#define checkCURAND(expression)                               \
  {                                                          \
    curandStatus_t status = (expression);                     \
    if (status != CURAND_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << _cudaGetErrorEnum(status) << std::endl;   \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


using namespace std;

int N_train = 60000, N_test = 10000;
int rows = 28, cols = 28, channels = 1;
int BW = 16 * 16;						// Block size for GPU kernel

// void roundUp(int a, int b) {

// }

__global__ void fillValue(float *v, int size, int value) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;
	v[i] = value;
}

__global__ void softmaxLossBackProp(float *y, float *SO, float *dSO, int batch_size, int output_size, float eps) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= batch_size)
		return;
	int cur_class = static_cast<int>(y[i]);
	dSO[i * output_size + cur_class] = -1 / (SO[i * output_size + cur_class] * batch_size + eps);
}

__global__ void inferClass(float *O, float *IO, int batch_size, int output_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= batch_size)
		return;

	float max = O[i * output_size];
	int index = 0;
	for (int j = 1; j < output_size; j++) {
		if (O[i * output_size + j] > max) {
			max = O[i * output_size + j];
			index = j;
		}
	}
	IO[i] = (float)index;
}

int reverseInt(int n) {
	int bytes = 4;
	unsigned char ch[bytes];
	for (int i = 0; i < bytes; i++) {
		ch[i] = (n >> i * 8) & 255;
	}
	int p = 0;
	for (int i = 0; i < bytes; i++) {
		p += (int) ch[i] << (bytes - i - 1) * 8;
	}
	return p;
}


void readMNIST(vector<vector<uchar> > &train_images, vector<vector<uchar> > &test_images, vector<uchar> &train_labels, vector<uchar> &test_labels) {
	string filename_train_images = "data/train-images.idx3-ubyte";
	string filename_train_labels = "data/train-labels.idx1-ubyte";

	string filename_test_images = "data/t10k-images.idx3-ubyte";
	string filename_test_labels = "data/t10k-labels.idx1-ubyte";

	// read train/test images
	for (int i = 0; i < 2; i++) {
		string filename;
		if (i == 0)
			filename = filename_train_images;
		else
			filename = filename_test_images;

		ifstream f(filename.c_str(), ios::binary);
		if (!f.is_open())
			printf("Cannot read MNIST from %s\n", filename.c_str());

		// read metadata
		int magic_number = 0, n_images = 0, n_rows = 0, n_cols = 0;
		f.read((char *) &magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		f.read((char *) &n_images, sizeof(n_images));
		n_images = reverseInt(n_images);
		f.read((char *) &n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		f.read((char *) &n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);

		for (int k = 0; k < n_images; k++) {
			vector<uchar> temp;
			temp.reserve(n_rows * n_cols);
			for (int j = 0; j < n_rows * n_cols; j++) {
				uchar t = 0;
				f.read((char *)&t, sizeof(t));
				temp.push_back(t);
			}
			if (i == 0)
				train_images.push_back(temp);
			else
				test_images.push_back(temp);
		}
		f.close();

	}

	// read train/test labels
	for (int i = 0; i < 2; i++) {
		string filename;
		if (i == 0)
			filename = filename_train_labels;
		else
			filename = filename_test_labels;

		ifstream f(filename.c_str(), ios::binary);
		if (!f.is_open())
			printf("Cannot read MNIST from %s\n", filename.c_str());

		// read metadata
		int magic_number = 0, n_labels = 0;
		f.read((char *) &magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		f.read((char *) &n_labels, sizeof(n_labels));
		n_labels = reverseInt(n_labels);

		for (int k = 0; k < n_labels; k++) {
			uchar t = 0;
			f.read((char *)&t, sizeof(t));
			if (i == 0)
				train_labels.push_back(t);
			else
				test_labels.push_back(t);
		}

		f.close();

	}


}

void printMatrix(float *M, int r, int c) {
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			cout << M[i * c + j] << ' ';
		}
		cout << endl;
	}
	cout << endl;
}

class Context {
public:	
	int batch_size, channels;
	int input_rows, input_cols, output_rows, output_cols;
	float learning_rate;
	float *X, *W1, *b1, *W2, *b2, *H, *Hrelu, *O, *SO, *dSO, *dO, *dW1, *db1, *dW2, *db2, *dH, *dHrelu;
	float *IO;
	float *y;
	float *onevec;
	float *h_IO;
	int input_size;
	int input_size_fc;
	int hidden_size;
	int output_size;
	int input_feature, output_feature;
	cublasHandle_t cublasHandle;
	// cudnnTensorDescriptor_t batchTensor, W1Tensor, b1Tensor, W2Tensor, b2Tensor, HTensor, OTensor;
	cudnnTensorDescriptor_t HTensor, OTensor;
	cudnnActivationDescriptor_t Reludesc;
	// cudnnOpTensorDescriptor_t Adddesc, Muldesc;

	cudnnHandle_t cudnnHandle;
	curandGenerator_t curandgen;

	float *h_W1, *h_W2, *h_b1, *h_b2, *h_SO, *h_y;
	float eps;

	// conv
	cudnnTensorDescriptor_t inputTensor, conv1OTensor, conv1bTensor;
	cudnnFilterDescriptor_t conv1Tensor;
	cudnnConvolutionDescriptor_t conv1Desc;
	cudnnConvolutionFwdAlgo_t conv1fAlgo;
	cudnnConvolutionBwdFilterAlgo_t conv1bfAlgo;
	size_t workspace_bytes;
	float *workspace;
	float *conv1O, *conv1OA;
	float *conv1filter, *conv1bias;
	float *dconv1filter, *dconv1bias;
	float *dconv1O, *dconv1OA;
	int filter_height, filter_width;
	
	// vdnn
	int req_algo_count;
	cudnnConvolutionFwdAlgoPerf_t *conv1fwdperf;
	cudnnConvolutionBwdFilterAlgoPerf_t *conv1bwdfperf;
	cudnnConvolutionBwdDataAlgoPerf_t *conv1bwddperf;

	cudnnPoolingDescriptor_t poolDesc;

	Context(int input_size, int batch_size, int hidden_size, float learning_rate, int output_size) {
		this->batch_size = batch_size;
		this->hidden_size = hidden_size;
		this->output_size = output_size;				// number of classes;
		this->channels = 1;
		
		input_rows = 28;
		input_cols = 28;
		input_feature = 256;
		output_rows = 28;
		output_cols = 28;
		output_feature = 128;
		filter_height = 1, filter_width = 1;
		this->input_size = input_rows * input_cols * input_feature;
		cout << "input_size: " << this->input_size << endl;
		input_size_fc = output_rows * output_cols * output_feature;
		this->learning_rate = learning_rate;
		eps = 1e-8;
		workspace_bytes = 0;
		workspace = NULL;


		checkCUBLAS(cublasCreate(&cublasHandle));
		checkCUDNN(cudnnCreate(&cudnnHandle));
		checkCURAND(curandCreateGenerator(&curandgen, CURAND_RNG_PSEUDO_DEFAULT));

		//vdnn
		req_algo_count = 10;
		conv1fwdperf = (cudnnConvolutionFwdAlgoPerf_t *)malloc(req_algo_count * sizeof(cudnnConvolutionFwdAlgoPerf_t));
		conv1bwdfperf = (cudnnConvolutionBwdFilterAlgoPerf_t *)malloc(req_algo_count * sizeof(cudnnConvolutionBwdFilterAlgoPerf_t));
		conv1bwddperf = (cudnnConvolutionBwdDataAlgoPerf_t *)malloc(req_algo_count * sizeof(cudnnConvolutionBwdDataAlgoPerf_t));

		// conv
		checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&conv1OTensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&conv1bTensor));
		checkCUDNN(cudnnCreateFilterDescriptor(&conv1Tensor));
		checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, input_feature, input_rows, input_cols));
		checkCUDNN(cudnnSetTensor4dDescriptor(conv1OTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, output_feature, output_rows, output_cols));
		checkCUDNN(cudnnSetTensor4dDescriptor(conv1bTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, output_feature, 1, 1));

		checkCUDNN(cudnnSetFilter4dDescriptor(conv1Tensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
											 output_feature, input_feature, filter_height, filter_width));

		checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1Desc));
		int pad_h = 0, pah_w = 0, u = 1, v = 1, dilation_h = 1, dilation_w = 1;
		checkCUDNN(cudnnSetConvolution2dDescriptor(conv1Desc, pad_h, pah_w, u, v, dilation_h, dilation_w,
													CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

		checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
		checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 
												filter_height, filter_width, pad_h, pad_w, u, v));

		int ret_algo_count;
		int n;
		// cout << "waiting..\n";
		// cin >> n;
		checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnnHandle, inputTensor, conv1Tensor, conv1Desc, conv1OTensor,
													 req_algo_count, &ret_algo_count, conv1fwdperf));
		cout << "Printing forward conv algo perf\n";
		for (int i = 0; i < ret_algo_count; i++) {
			cout << i << endl;
			cout << "algo: " << conv1fwdperf[i].algo << endl;
			cout << "status: " << cudnnGetErrorString(conv1fwdperf[i].status) << endl;
			cout << "time(ms): " << conv1fwdperf[i].time << endl;
			cout << "memory(bytes): " << conv1fwdperf[i].memory << endl;
			cout << "mathType: " << conv1fwdperf[i].mathType << endl;
			cout << endl;
		}

		checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle, inputTensor, conv1OTensor, conv1Desc, conv1Tensor,
																req_algo_count, &ret_algo_count, conv1bwdfperf));

		cout << "Printing bwdfilter conv algo perf\n";
		for (int i = 0; i < ret_algo_count; i++) {
			cout << i << endl;
			cout << "algo: " << conv1bwdfperf[i].algo << endl;
			cout << "status: " << cudnnGetErrorString(conv1bwdfperf[i].status) << endl;
			cout << "time(ms): " << conv1bwdfperf[i].time << endl;
			cout << "memory(bytes): " << conv1bwdfperf[i].memory << endl;
			cout << "mathType: " << conv1bwdfperf[i].mathType << endl;
			cout << endl;
		}

		checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle, conv1Tensor, conv1OTensor, conv1Desc, inputTensor,
																req_algo_count, &ret_algo_count, conv1bwddperf));

		cout << "Printing bwddata conv algo perf\n";
		for (int i = 0; i < ret_algo_count; i++) {
			cout << i << endl;
			cout << "algo: " << conv1bwdfperf[i].algo << endl;
			cout << "status: " << cudnnGetErrorString(conv1bwdfperf[i].status) << endl;
			cout << "time(ms): " << conv1bwdfperf[i].time << endl;
			cout << "memory(bytes): " << conv1bwdfperf[i].memory << endl;
			cout << "mathType: " << conv1bwdfperf[i].mathType << endl;
			cout << endl;
		}

		// checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, inputTensor, conv1Tensor, conv1Desc, conv1OTensor,
		// 												CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv1fAlgo));
		// size_t fwd_wspace;
		// checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputTensor, conv1Tensor, conv1Desc, conv1OTensor,
		// 													conv1fAlgo, &fwd_wspace));
		// cout << "fwd_wspace: " << fwd_wspace << endl;
		// cout << "algo_used: " << conv1fAlgo << endl;
		// // exit(0);
		// workspace_bytes = max(workspace_bytes, fwd_wspace);

		// checkCudaErrors(cudaMalloc((void **)&conv1filter, filter_height * filter_width * input_feature * output_feature * sizeof(float) + sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&dconv1filter, filter_height * filter_width * input_feature * output_feature * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&conv1bias, output_feature * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&dconv1bias, output_feature * sizeof(float)));
		// fillValue<<<ceil(1.0 * output_feature / BW), BW>>>(conv1bias, output_feature, 0);
		// checkCURAND(curandGenerateNormal(curandgen, conv1filter, filter_height * filter_width * input_feature * output_feature + 1,
		// 								0, 1 / sqrt(filter_height * filter_width * input_feature)));

		// checkCudaErrors(cudaMalloc((void **)&conv1O, batch_size * input_size_fc * sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&conv1OA, batch_size * this->input_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&dconv1O, batch_size * input_size_fc * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&dconv1OA, batch_size * input_size_fc * sizeof(float)));

		// checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle, inputTensor, conv1OTensor, conv1Desc, conv1Tensor,
		// 												CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &conv1bfAlgo));

		// size_t bwd_fwspace;
		// checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, inputTensor, conv1OTensor, conv1Desc, conv1Tensor,
		// 															conv1bfAlgo, &bwd_fwspace));
		// workspace_bytes = max(workspace_bytes, bwd_fwspace);
		// if (workspace_bytes > 0)
		// 	checkCudaErrors(cudaMalloc((void **)&workspace, workspace_bytes));


		// // allocate memory in device
		checkCudaErrors(cudaMalloc((void **)&X, batch_size * this->input_size * sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&OP, batch_size * this->input_size_fc * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&W1, input_size_fc * hidden_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&dW1, input_size_fc * hidden_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&b1, hidden_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&db1, hidden_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&W2, hidden_size * output_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&dW2, hidden_size * output_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&b2, output_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&db2, output_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&H, batch_size * hidden_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&dH, batch_size * hidden_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&Hrelu, batch_size * hidden_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&dHrelu, batch_size * hidden_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&O, batch_size * output_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&IO, batch_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&dO, batch_size * output_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&SO, batch_size * output_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&dSO, batch_size * output_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&y, batch_size * sizeof(float)));
		// checkCudaErrors(cudaMalloc((void **)&onevec, batch_size * sizeof(float)));
		// fillValue<<<ceil(1.0 * batch_size / BW), BW>>>(onevec, batch_size, 1);

		// // {
		// // 	cout << "waiting..\n";
		// // 	int n;
		// // 	cin >> n;
		// // }
		// // h_IO = (float *)malloc(batch_size * sizeof(float));

		// // checkCUDNN(cudnnCreateTensorDescriptor(&batchTensor));
		// // checkCUDNN(cudnnCreateTensorDescriptor(&W1Tensor));
		// // checkCUDNN(cudnnCreateTensorDescriptor(&b1Tensor));
		// // checkCUDNN(cudnnCreateTensorDescriptor(&W2Tensor));
		// // checkCUDNN(cudnnCreateTensorDescriptor(&b2Tensor));
		// checkCUDNN(cudnnCreateTensorDescriptor(&HTensor));
		// checkCUDNN(cudnnCreateTensorDescriptor(&OTensor));
		// checkCUDNN(cudnnCreateActivationDescriptor(&Reludesc));
		// // checkCUDNN(cudnnCreateOpTensorDescriptor(&Opdesc));


		// // checkCUDNN(cudnnSetTensor4dDescriptor(batchTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, rows * cols, 1, 1));

		// // // just to be able to multiply properly
		// // checkCUDNN(cudnnSetTensor4dDescriptor(W1Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, hidden_size, 1, 1));
		// // checkCUDNN(cudnnSetTensor4dDescriptor(b1Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, hidden_size, 1, 1));

		// checkCUDNN(cudnnSetTensor4dDescriptor(HTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, hidden_size, 1, 1));

		// // // just to be able to multiply properly
		// // checkCUDNN(cudnnSetTensor4dDescriptor(W2Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, hidden_size, output_size, 1, 1));
		// // checkCUDNN(cudnnSetTensor4dDescriptor(b2Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, output_size, 1, 1));

		// checkCUDNN(cudnnSetTensor4dDescriptor(OTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, output_size, 1, 1));

		// checkCUDNN(cudnnSetActivationDescriptor(Reludesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));

		// // initialization
		// fillValue<<<ceil(1.0 * hidden_size / BW), BW>>>(b1, hidden_size, 0);
		// fillValue<<<ceil(1.0 * output_size / BW), BW>>>(b2, output_size, 0);


		// checkCURAND(curandGenerateNormal(curandgen, W1, input_size_fc * hidden_size, 0, 0.1));
		// checkCURAND(curandGenerateNormal(curandgen, W2, hidden_size * output_size, 0, 0.1));

		checkCURAND(curandGenerateNormal(curandgen, conv1OA, batch_size * this->input_size, 0, 0.1));

		// h_W1 = (float *)malloc(input_size_fc * hidden_size * sizeof(float));
		// h_W2 = (float *)malloc(hidden_size * output_size * sizeof(float));
		// h_b1 = (float *)malloc(hidden_size * sizeof(float));
		// h_b2 = (float *)malloc(output_size * sizeof(float));
		// h_SO = (float *)malloc(batch_size * output_size * sizeof(float));
		// h_y = (float *)malloc(batch_size * sizeof(float));

		


	}

	~Context() {
		checkCUBLAS(cublasDestroy(cublasHandle));
		checkCURAND(curandDestroyGenerator(curandgen));

		// checkCudaErrors(cudaFree(X));
		// checkCudaErrors(cudaFree(W1));
		// checkCudaErrors(cudaFree(dW1));
		// checkCudaErrors(cudaFree(b1));
		// checkCudaErrors(cudaFree(db1));
		// checkCudaErrors(cudaFree(W2));
		// checkCudaErrors(cudaFree(dW2));
		// checkCudaErrors(cudaFree(b2));
		// checkCudaErrors(cudaFree(db2));
		// checkCudaErrors(cudaFree(H));
		// checkCudaErrors(cudaFree(dH));
		// checkCudaErrors(cudaFree(Hrelu));
		// checkCudaErrors(cudaFree(dHrelu));
		// checkCudaErrors(cudaFree(O));
		// checkCudaErrors(cudaFree(dO));
		// checkCudaErrors(cudaFree(SO));
		// checkCudaErrors(cudaFree(dSO));
		// checkCudaErrors(cudaFree(IO));
		// free(h_IO);
		// checkCUDNN(cudnnDestroyActivationDescriptor(Reludesc));
		// // checkCUDNN(cudnnDestroyTensorDescriptor(batchTensor));
		// // checkCUDNN(cudnnDestroyTensorDescriptor(W1Tensor));
		// // checkCUDNN(cudnnDestroyTensorDescriptor(b1Tensor));
		// // checkCUDNN(cudnnDestroyTensorDescriptor(W2Tensor));
		// // checkCUDNN(cudnnDestroyTensorDescriptor(b2Tensor));
		// checkCUDNN(cudnnDestroyTensorDescriptor(HTensor));
		// checkCUDNN(cudnnDestroyTensorDescriptor(OTensor));
		// checkCUDNN(cudnnDestroy(cudnnHandle));

		

		// free(h_W1);
		// free(h_W2);
		// free(h_b1);
		// free(h_b2);
		// free(h_SO);
		// free(h_y);

		// // conv
		// checkCUDNN(cudnnDestroyFilterDescriptor(conv1Tensor));

	}

	void forwardPropagate(bool train=true) {
		// float alpha = 1.0f, beta = 0.0f;
		
		// conv
		// conv forward
		cudaEvent_t start, stop;
		float milli = 0;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		// // checkCudaErrors(cudaEventRecord(start));
		// // checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, inputTensor, X, conv1Tensor, conv1filter, conv1Desc,
		// // 									conv1fAlgo, workspace, workspace_bytes, &beta, conv1OTensor, conv1O));
		// // checkCudaErrors(cudaEventRecord(stop));
		// // checkCudaErrors(cudaEventSynchronize(stop));
		// // checkCudaErrors(cudaEventElapsedTime(&milli, start, stop));
		// // cout << "time for conv: " << milli << endl;
		// // // add bias
		// // checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv1bTensor, conv1bias, &alpha, conv1OTensor, conv1O));

		// // // activation
		// // checkCUDNN(cudnnActivationForward(cudnnHandle, Reludesc, &alpha, conv1OTensor, conv1O, &beta, conv1OTensor, conv1OA));

		float *temp = NULL;
		checkCudaErrors(cudaMallocHost((void **)&temp, batch_size * input_size * sizeof(float)));
		checkCudaErrors(cudaEventRecord(start));
		checkCudaErrors(cudaMemcpyAsync(temp, conv1OA, batch_size * input_size * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaEventRecord(stop));
		cudaEventSynchronize(stop);
		checkCudaErrors(cudaEventElapsedTime(&milli, start, stop));
		cout << "transfer time(ms): " << milli << endl;
		int n;
		cin >> n;
		for (int i = 0; i < n; i++) {
			cout << temp[i];
		}

		checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, inputTensor, X, &beta, conv1OTensor, OP))
		// int n;
		// cout << "waiting..\n";
		// cin >> n;
		exit(0);

		// // multiply weights to input
		// checkCudaErrors(cudaEventRecord(start));
		// checkCUBLAS(cublasSgemm(cublasHandle, 
		// 						CUBLAS_OP_N, CUBLAS_OP_N,
		// 						hidden_size, batch_size, input_size_fc,
		// 						&alpha,
		// 						W1, hidden_size,
		// 						conv1OA, input_size_fc,
		// 						&beta,
		// 						H, hidden_size));
		// checkCudaErrors(cudaEventRecord(stop));
		// checkCudaErrors(cudaEventSynchronize(stop));
		// checkCudaErrors(cudaEventElapsedTime(&milli, start, stop));
		// cout << "time for mul: " << milli << endl;
		// // exit(0);
		// // float *h_X = (float *)malloc(batch_size * input_size * sizeof(float));
		// // checkCudaErrors(cudaMemcpy(h_X, X, batch_size * input_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // cout << "X:\n";
		// // printMatrix(h_X, batch_size, input_size);
		// // int n;
		// // cout << "waiting..\n";
		// // cin >> n;
		
		// // checkCudaErrors(cudaMemcpy(h_W1, W1, input_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // cout << "W1:\n";
		// // printMatrix(h_W1, input_size, hidden_size);
		// // cout << "waiting..\n";
		// // cin >> n;

		// // float *h_H = (float *)malloc(batch_size * hidden_size * sizeof(float));
		// // checkCudaErrors(cudaMemcpy(h_H, H, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // cout << "H:\n";
		// // printMatrix(h_H, batch_size, hidden_size);
		// // cout << "waiting..\n";
		// // cin >> n;

		// // add bias to output
		// checkCUBLAS(cublasSgemm(cublasHandle,
		// 						CUBLAS_OP_N, CUBLAS_OP_N,
		// 						hidden_size, batch_size, 1,
		// 						&alpha,
		// 						b1, hidden_size,
		// 						onevec, 1,
		// 						&alpha,
		// 						H, hidden_size));

		// // checkCudaErrors(cudaMemcpy(h_b1, b1, hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // cout << "b1:\n";
		// // printMatrix(h_b1, 1, hidden_size);
		// // cout << "waiting..\n";
		// // cin >> n;

		// // checkCudaErrors(cudaMemcpy(h_H, H, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // cout << "H+b:\n";
		// // printMatrix(h_H, batch_size, hidden_size);
		// // cout << "waiting..\n";
		// // cin >> n;

		// // apply relu activation
		// checkCUDNN(cudnnActivationForward(cudnnHandle, Reludesc, &alpha, HTensor, H, &beta, HTensor, Hrelu));

		// // checkCudaErrors(cudaMemcpy(h_H, Hrelu, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // cout << "Hrelu:\n";
		// // printMatrix(h_H, batch_size, hidden_size);
		// // cout << "waiting..\n";
		// // cin >> n;

		// // multiply weights to input
		// checkCUBLAS(cublasSgemm(cublasHandle, 
		// 						CUBLAS_OP_N, CUBLAS_OP_N,
		// 						output_size, batch_size, hidden_size,
		// 						&alpha,
		// 						W2, output_size,
		// 						Hrelu, hidden_size,
		// 						&beta,
		// 						O, output_size));
		

		// // checkCudaErrors(cudaMemcpy(h_W2, W2, hidden_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // cout << "W2:\n";
		// // printMatrix(h_W2, hidden_size, output_size);
		// // cout << "waiting..\n";
		// // cin >> n;


		// // float *h_O = (float *)malloc(batch_size * output_size * sizeof(float));
		// // checkCudaErrors(cudaMemcpy(h_O, O, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // cout << "O:\n";
		// // printMatrix(h_O, batch_size, output_size);
		// // cout << "waiting..\n";
		// // cin >> n;

		// // add bias to output
		// checkCUBLAS(cublasSgemm(cublasHandle,
		// 						CUBLAS_OP_N, CUBLAS_OP_N,
		// 						output_size, batch_size, 1,
		// 						&alpha,
		// 						b2, output_size,
		// 						onevec, 1,
		// 						&alpha,
		// 						O, output_size));

		// // checkCudaErrors(cudaMemcpy(h_b2, b2, output_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // cout << "b2:\n";
		// // printMatrix(h_b2, 1, output_size);
		// // cout << "waiting..\n";
		// // cin >> n;

		// // checkCudaErrors(cudaMemcpy(h_O, O, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // cout << "O+b:\n";
		// // printMatrix(h_O, batch_size, output_size);
		// // cout << "waiting..\n";
		// // cin >> n;

		// if (train == false) {
		// 	inferClass<<<ceil(1.0 * batch_size / BW), BW>>>(O, IO, batch_size, output_size);
		// 	cudaMemcpy(h_IO, IO, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
		// 	return;
		// }

		// // float *sm_test = (float *)malloc(batch_size * output_size * sizeof(float));
		// // for (int i = 0; i < batch_size; i++) {
		// // 	for (int j = 0; j < output_size; j++) {
		// // 		sm_test[i * output_size + j] = i * output_size + j;
		// // 	}
		// // }
		// // cout << "sm_test:\n";
		// // printMatrix(sm_test, batch_size, output_size);
		// // checkCudaErrors(cudaMemcpy(O, sm_test, batch_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
		// // checkCudaErrors(cudaMemcpy(sm_test, O, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // cout << "O:\n";
		// // printMatrix(sm_test, batch_size, output_size);
		// // apply softmax
		// // checkCudaErrors(cudaMemcpy(h_SO, O, output_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // printMatrix(h_SO, batch_size, output_size);
		// checkCudaErrors(cudaDeviceSynchronize());
		// checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, OTensor, O, &beta, OTensor, SO));
		// checkCudaErrors(cudaDeviceSynchronize());
		// // checkCudaErrors(cudaMemcpy(h_SO, SO, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // cout << "SO:\n";
		// // printMatrix(h_SO, batch_size, output_size);
		// // cout << "waiting..\n";
		// // cin >> n;
		
		// // checkCudaErrors(cudaMemcpy(h_SO, SO, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // checkCudaErrors(cudaMemcpy(h_y, y, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
		// // float loss = 0;
		// // for (int i = 0; i < batch_size; i++)
		// // 	loss += -log(h_SO[i * output_size + int(h_y[i])]);
		// // // getSoftmaxLoss<<<ceil(1.0 * batch_size / BW), BW>>>(SO, y, batch_size, output_size, &loss);
		// // // printf("yay");
		// // return loss;

	}

	// void backwardPropagate() {
	// 	float alpha = 1.0f, beta = 0.0f;
	// 	checkCudaErrors(cudaMemset(dSO, 0, batch_size * output_size * sizeof(float)));
	// 	softmaxLossBackProp<<<ceil(1.0 * batch_size / BW), BW>>>(y, SO, dSO, batch_size, output_size, eps);
	// 	// int n;
	// 	// checkCudaErrors(cudaMemcpy(h_SO, dSO, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
	// 	// cout << "dSO:\n";
	// 	// printMatrix(h_SO, batch_size, output_size);
	// 	// cout << "waiting..\n";
	// 	// cin >> n;

	// 	// softmax backprop
	// 	checkCUDNN(cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
	// 									OTensor, SO, OTensor, dSO,
	// 									&beta,
	// 									OTensor, dO));

	// 	// checkCudaErrors(cudaMemcpy(h_SO, dO, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
	// 	// cout << "dO:\n";
	// 	// printMatrix(h_SO, batch_size, output_size);
	// 	// cout << "waiting..\n";
	// 	// cin >> n;
		
	// 	// gradient w.r.t. b2
	// 	checkCUBLAS(cublasSgemm(cublasHandle, 
	// 							CUBLAS_OP_N, CUBLAS_OP_N,
	// 							output_size, 1, batch_size,
	// 							&alpha,
	// 							dO, output_size,
	// 							onevec, batch_size,
	// 							&beta,
	// 							db2, output_size));

	// 	// checkCudaErrors(cudaMemcpy(h_b2, db2, output_size * sizeof(float), cudaMemcpyDeviceToHost));
	// 	// cout << "db2:\n";
	// 	// printMatrix(h_b2, 1, output_size);
	// 	// cout << "waiting..\n";
	// 	// cin >> n;

	// 	// checkCudaErrors(cudaMemcpy(h_b2, db2, output_size * sizeof(float), cudaMemcpyDeviceToHost));
	// 	// printMatrix(h_b2, 1, output_size);
		
	// 	checkCudaErrors(cudaDeviceSynchronize());


	// 	// gradient w.r.t. W2
	// 	checkCUBLAS(cublasSgemm(cublasHandle, 
	// 							CUBLAS_OP_N, CUBLAS_OP_T,
	// 							output_size, hidden_size, batch_size,
	// 							&alpha,
	// 							dO, output_size,
	// 							Hrelu, hidden_size,
	// 							&beta,
	// 							dW2, output_size));

	// 	// checkCudaErrors(cudaMemcpy(h_W2, dW2, hidden_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
	// 	// cout << "dW2:\n";
	// 	// printMatrix(h_W2, hidden_size, output_size);
	// 	// cout << "waiting..\n";
	// 	// cin >> n;		

	// 	// gradient w.r.t. Hrelu
	// 	checkCUBLAS(cublasSgemm(cublasHandle,
	// 							CUBLAS_OP_T, CUBLAS_OP_N,
	// 							hidden_size, batch_size, output_size,
	// 							&alpha,
	// 							W2, output_size,
	// 							dO, output_size,
	// 							&beta,
	// 							dHrelu, hidden_size));

	// 	// float *h_H = (float *)malloc(batch_size * hidden_size * sizeof(float));
	// 	// checkCudaErrors(cudaMemcpy(h_H, dHrelu, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
	// 	// cout << "dHrelu:\n";
	// 	// printMatrix(h_H, batch_size, hidden_size);
	// 	// cout << "waiting..\n";
	// 	// cin >> n;

	// 	// gradient w.r.t H
	// 	checkCUDNN(cudnnActivationBackward(cudnnHandle, Reludesc, &alpha, HTensor, Hrelu, HTensor, dHrelu,
	// 										HTensor, H, &beta, HTensor, dH));


	// 	// gradient w.r.t. b1
	// 	checkCUBLAS(cublasSgemm(cublasHandle, 
	// 							CUBLAS_OP_N, CUBLAS_OP_N,
	// 							hidden_size, 1, batch_size,
	// 							&alpha,
	// 							dH, hidden_size,
	// 							onevec, batch_size,
	// 							&beta,
	// 							db1, hidden_size));

	// 	// gradient w.r.t. W1
	// 	checkCUBLAS(cublasSgemm(cublasHandle, 
	// 							CUBLAS_OP_N, CUBLAS_OP_T,
	// 							hidden_size, input_size_fc, batch_size,
	// 							&alpha,
	// 							dH, hidden_size,
	// 							conv1OA, input_size_fc,
	// 							&beta,
	// 							dW1, hidden_size));

	// 	// gradient w.r.t. conv1OA
	// 	checkCUBLAS(cublasSgemm(cublasHandle,
	// 							CUBLAS_OP_T, CUBLAS_OP_N,
	// 							input_size_fc, batch_size, hidden_size,
	// 							&alpha,
	// 							W1, hidden_size,
	// 							dH, hidden_size,
	// 							&beta,
	// 							dconv1OA, input_size_fc));

	// 	// gradient w.r.t conv1O
	// 	checkCUDNN(cudnnActivationBackward(cudnnHandle, Reludesc, &alpha, conv1OTensor, conv1OA, conv1OTensor, dconv1OA,
	// 										conv1OTensor, conv1O, &beta, conv1OTensor, dconv1O));

	// 	// gradient w.r.t. conv1bias
	// 	checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv1OTensor, dconv1O, &beta, conv1bTensor, dconv1bias));

	// 	// gradient w.r.t. conv1filter
	// 	checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, inputTensor, X, conv1OTensor, dconv1O, conv1Desc,
	// 												conv1bfAlgo, workspace, workspace_bytes, &beta, conv1Tensor, dconv1filter));



	// }

	// void updateWeights() {
	// 	float alpha = -learning_rate;

	// 	// update W1
	// 	checkCUBLAS(cublasSaxpy(cublasHandle, input_size * hidden_size,
	// 							&alpha,
	// 							dW1, 1,
	// 							W1, 1));

	// 	//update b1
	// 	checkCUBLAS(cublasSaxpy(cublasHandle, hidden_size,
	// 							&alpha,
	// 							db1, 1,
	// 							b1, 1));

	// 	// update W2
	// 	checkCUBLAS(cublasSaxpy(cublasHandle, hidden_size * output_size,
	// 							&alpha,
	// 							dW2, 1,
	// 							W2, 1));
	// 	//update b2
	// 	checkCUBLAS(cublasSaxpy(cublasHandle, output_size,
	// 							&alpha,
	// 							db2, 1,
	// 							b2, 1));

	// 	// update conv1bias
	// 	checkCUBLAS(cublasSaxpy(cublasHandle, output_feature,
	// 							&alpha,
	// 							dconv1bias, 1,
	// 							conv1bias, 1));

	// 	// update conv1filter
	// 	checkCUBLAS(cublasSaxpy(cublasHandle, output_feature * input_feature * filter_height * filter_width,
	// 							&alpha,
	// 							dconv1filter, 1,
	// 							conv1filter, 1));

	// 	// checkCudaErrors(cudaMemcpy(h_W1, W1, hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
	// 	// 	for (int i = 0; i < output_size; i++) {
	// 	// 		cout << h_W2[i] << ' ';
	// 	// 	}
	// 	// 	cout << endl;
	// 	// 	checkCudaErrors(cudaDeviceSynchronize());

	// }

	void train(int num_iter, float *train_images, float *train_labels, float *test_images, float *test_labels, int N) {
		// int image_size = rows * cols * channels;

		for (int iter = 0; iter < num_iter; iter++) {
			int image_id = iter % (N / batch_size);

			// checkCudaErrors(cudaMemcpy(h_W1, W1, input_size_fc * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
			// checkCudaErrors(cudaMemcpy(h_W2, W2, hidden_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
			// checkCudaErrors(cudaMemcpy(h_b1, b1, hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
			// checkCudaErrors(cudaMemcpy(h_b2, b2, output_size * sizeof(float), cudaMemcpyDeviceToHost));
			
			// checkCudaErrors(cudaMemcpy(X, &train_images[image_id * batch_size * input_size], input_size * batch_size * sizeof(float), cudaMemcpyHostToDevice));
			// checkCudaErrors(cudaMemcpy(y, &train_labels[image_id * batch_size], batch_size * sizeof(float), cudaMemcpyHostToDevice));

			this->forwardPropagate();
			// this->backwardPropagate();
			// this->updateWeights();

			checkCudaErrors(cudaDeviceSynchronize());
			exit(0);
			// checkCudaErrors(cudaMemcpy(h_W2, W2, hidden_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
			// for (int i = 0; i < output_size; i++) {
			// 	cout << h_W2[i] << ' ';
			// }
			// cout << endl;
			checkCudaErrors(cudaDeviceSynchronize());
		}
	}

	int test(float *test_images, float *test_labels, int N) {
		// int image_size = rows * cols * channels;
		int start = 0;
		int size = batch_size;
		int count = 0;
		while (start < N) {
			if (start + size >= N)
				size = N - start;
			checkCudaErrors(cudaMemcpy(X, &test_images[start * input_size], input_size * size * sizeof(float), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(y, &test_labels[start], size * sizeof(float), cudaMemcpyHostToDevice));
			this->forwardPropagate(false);
			checkCudaErrors(cudaDeviceSynchronize());
			for (int i = 0; i < size; i++) {
				if (h_IO[i] == test_labels[start + i])
					count++;
				// cout << h_IO[i] << ' ';
			}
			start = start + size;
		}
		return count;

	}


};

int main() {
	vector<vector<uchar> > train_images, test_images;
	vector<uchar> train_labels, test_labels;
	readMNIST(train_images, test_images, train_labels, test_labels);
	float *f_train_images, *f_train_labels, *f_test_images, *f_test_labels;
	int input_size = rows * cols * channels;
	f_train_images = (float *)malloc(N_train * input_size * sizeof(float));
	f_train_labels = (float *)malloc(N_train * sizeof(float));
	f_test_images = (float *)malloc(N_test * input_size * sizeof(float));
	f_test_labels = (float *)malloc(N_test * sizeof(float));
	// checkCudaErrors(cudaMallocHost((void **)&f_train_images, N_train * input_size * sizeof(float)));
	// checkCudaErrors(cudaMallocHost((void **)&f_train_labels, N_train * sizeof(float)));
	// checkCudaErrors(cudaMallocHost((void **)&f_test_images, N_test * input_size * sizeof(float)));
	// checkCudaErrors(cudaMallocHost((void **)&f_test_labels, N_test * sizeof(float)));

	float *mean_image;
	mean_image = (float *)malloc(input_size * sizeof(float));

	for (int k = 0; k < N_train; k++) {
		for (int j = 0; j < rows * cols; j++) {
			f_train_images[k * input_size + j] = (float)train_images[k][j];
		}
		f_train_labels[k] = (float)train_labels[k];
	}

	for (int k = 0; k < N_test; k++) {
		for (int j = 0; j < rows * cols; j++) {
			f_test_images[k * input_size + j] = (float)test_images[k][j];
		}
		f_test_labels[k] = (float)test_labels[k];
	}

	for (int i = 0; i < input_size; i++) {
		mean_image[i] = 0;
		for (int k = 0; k < N_train; k++) {
			mean_image[i] += f_train_images[k * input_size + i];
		}
		mean_image[i] /= N_train;
	}

	for (int i = 0; i < N_train; i++) {
		for (int j = 0; j < input_size; j++)
			f_train_images[i * input_size + j] -= mean_image[j];
	}

	for (int i = 0; i < N_test; i++) {
		for (int j = 0; j < input_size; j++)
			f_test_images[i * input_size + j] -= mean_image[j];
	}


	// int toy_input_size = 2;
	// int toy_hidden_size = 5;
	// int toy_output_size = 2;
	// int batch_size = 100;
	// float *toy_train, *toy_train_labels;
	// toy_train = (float *)malloc(batch_size * toy_input_size * sizeof(float));
	// toy_train_labels = (float *)malloc(batch_size * sizeof(float));
	// curandGenerator_t curandgen;
	// checkCURAND(curandCreateGeneratorHost(&curandgen, CURAND_RNG_PSEUDO_DEFAULT));
	// printf("toy_train, before init:\n");
	// printMatrix(toy_train, batch_size, toy_input_size);
	// checkCURAND(curandGenerateNormal(curandgen, toy_train, batch_size * toy_input_size * sizeof(float), 0, 10));
	// printf("toy_train, after init:\n");
	// printMatrix(toy_train, batch_size, toy_input_size);
	// for (int i = 0; i < batch_size; i++) {
	// 	cout << float(i % 2) << " p\n";
	// 	toy_train_labels[i] = float(i % 2);
	// }
	// printf("toy_train_labels, after init\n");
	// printMatrix(toy_train_labels, batch_size, 1);
	// int n;
	// cin >> n;
 
	// float toy_l_rate = 1e-1;
	// Context context(toy_input_size, batch_size, toy_hidden_size, toy_l_rate, toy_output_size);
	// int n_iter = 100;
	// int n_rep = 10;
	// for (int i = 0; i < n_rep; i++) {
	// 	context.train(n_iter, toy_train, toy_train_labels, toy_train, toy_train_labels, batch_size);
	// 	cout << context.test(toy_train, toy_train_labels, batch_size) << endl << flush;
	// }

	float l_rate = 1e-3;
	int hidden_size = 50;
	int batch_size = 128;
	int output_size = 10;
	Context context(input_size, batch_size, hidden_size, l_rate, output_size);
	int n_iter = 10000;
	int n_rep = 10;


	for (int i = 0; i < n_rep; i++) {
		context.train(n_iter, f_train_images, f_train_labels, f_test_images, f_test_labels, N_train);
		cout << context.test(f_test_images, f_test_labels, N_test) << endl;
	}


}