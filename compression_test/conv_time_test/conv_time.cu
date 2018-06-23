#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>

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

	cudnnHandle_t cudnn_handle;
	curandGenerator_t curandgen;

	float *h_W1, *h_W2, *h_b1, *h_b2, *h_SO, *h_y;
	float eps;

	// conv
	cudnnTensorDescriptor_t input_tensor, output_tensor, bias_tensor, pooling_output_tensor;
	cudnnFilterDescriptor_t filter_desc;
	cudnnConvolutionDescriptor_t conv_desc;
	cudnnActivationDescriptor_t actv_desc;
	cudnnPoolingDescriptor_t pool_desc;
	cudnnConvolutionFwdAlgo_t conv_fwd_algo;
	cudnnConvolutionBwdFilterAlgo_t conv_bwdf_algo;
	size_t workspace_size;
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

	Context(int input_size, int batch_size, int hidden_size, float learning_rate, int output_size, int filter_size) {
		this->batch_size = batch_size;
		this->hidden_size = hidden_size;
		this->output_size = output_size;				// number of classes;
		this->channels = 1;
		
		this->batch_size = batch_size = 128;
		input_rows = 112;
		input_cols = 112;
		input_feature = 128;
		output_rows = 112;
		output_cols = 112;
		output_feature = 128;
		filter_height = filter_size, filter_width = filter_size;
		int pad_h = filter_size / 2, pad_w = filter_size / 2, u = 1, v = 1, dilation_h = 1, dilation_w = 1;
		this->input_size = input_rows * input_cols * input_feature;
		cout << "input_size: " << this->input_size << endl;
		// input_size_fc = output_rows * output_cols * output_feature;
		this->learning_rate = learning_rate;
		eps = 1e-8;
		workspace_size = 0;
		workspace = NULL;

		// find time for conv or pool
		bool conv_test = false;

		checkCUBLAS(cublasCreate(&cublasHandle));
		checkCUDNN(cudnnCreate(&cudnn_handle));
		checkCURAND(curandCreateGenerator(&curandgen, CURAND_RNG_PSEUDO_DEFAULT));

		//vdnn
		req_algo_count = 10;
		conv1fwdperf = (cudnnConvolutionFwdAlgoPerf_t *)malloc(req_algo_count * sizeof(cudnnConvolutionFwdAlgoPerf_t));
		conv1bwdfperf = (cudnnConvolutionBwdFilterAlgoPerf_t *)malloc(req_algo_count * sizeof(cudnnConvolutionBwdFilterAlgoPerf_t));
		conv1bwddperf = (cudnnConvolutionBwdDataAlgoPerf_t *)malloc(req_algo_count * sizeof(cudnnConvolutionBwdDataAlgoPerf_t));

		// conv
		checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&output_tensor));
		checkCUDNN(cudnnCreateTensorDescriptor(&bias_tensor));
		checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
		checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, input_feature, input_rows, input_cols));
		checkCUDNN(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, output_feature, output_rows, output_cols));
		checkCUDNN(cudnnSetTensor4dDescriptor(bias_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, output_feature, 1, 1));

		checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
											 output_feature, input_feature, filter_height, filter_width));

		checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
		
		checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, u, v, dilation_h, dilation_w,
													CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

		checkCUDNN(cudnnCreateActivationDescriptor(&actv_desc));
		checkCUDNN(cudnnSetActivationDescriptor(actv_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 1e-8));

		int pooling_output_rows, pooling_output_cols;
		if (!conv_test) {
			checkCUDNN(cudnnCreatePoolingDescriptor(&pool_desc));
			u = v = 2;
			filter_height = filter_width = 2;
			pad_h = pad_w = 0;
			pooling_output_rows = (input_rows + 2 * pad_w - filter_width) / u, pooling_output_cols = (input_cols + 2 * pad_h - filter_height) / v;
			checkCUDNN(cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
													filter_height, filter_width,
													pad_h, pad_w,
													u, v));
			checkCUDNN(cudnnCreateTensorDescriptor(&pooling_output_tensor));
			checkCUDNN(cudnnSetTensor4dDescriptor(pooling_output_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, input_feature, pooling_output_rows, pooling_output_cols));
		}

		int ret_algo_count;
		int n;
		// cout << "waiting..\n";
		// cin >> n;
		if (conv_test) {
			checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn_handle, input_tensor, filter_desc, conv_desc, output_tensor,
															 req_algo_count, &ret_algo_count, conv1fwdperf));
			cerr << "Printing forward conv algo perf\n";
			cerr << "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM " << CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM << endl;
			cerr << "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM " << CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM << endl;
			cerr << "CUDNN_CONVOLUTION_FWD_ALGO_GEMM " << CUDNN_CONVOLUTION_FWD_ALGO_GEMM << endl;
			cerr << "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT " << CUDNN_CONVOLUTION_FWD_ALGO_DIRECT << endl;
			cerr << "CUDNN_CONVOLUTION_FWD_ALGO_FFT " << CUDNN_CONVOLUTION_FWD_ALGO_FFT << endl;
			cerr << "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING " << CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING << endl;
			cerr << "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD " << CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD << endl;
			cerr << "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED " << CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED << endl;
			for (int i = 0; i < ret_algo_count; i++) {
				cerr << i << endl;
				cerr << "algo: " << conv1fwdperf[i].algo << endl;
				cerr << "status: " << cudnnGetErrorString(conv1fwdperf[i].status) << endl;
				cerr << "time(ms): " << conv1fwdperf[i].time << endl;
				cerr << "memory(bytes): " << conv1fwdperf[i].memory << endl;
				cerr << "mathType: " << conv1fwdperf[i].mathType << endl;
				cerr << endl;
			}
			conv_fwd_algo = conv1fwdperf[0].algo;
			workspace_size = conv1fwdperf[0].memory;
		}

		// {
		// 	int n;
		// 	cout << "waiting..\n";
		// 	cin >> n;
		// }
		float alpha = 1.0, beta = 0.0;

		void *layer_input, *layer_output, *workspace, *W, *b, *pooling_layer_output;
		checkCudaErrors(cudaMalloc(&layer_input, batch_size * input_feature * input_rows * input_cols * sizeof(float)));
		checkCudaErrors(cudaMalloc(&layer_output, batch_size * output_feature * output_rows * output_cols * sizeof(float)));
		checkCudaErrors(cudaMalloc(&W, output_feature * input_feature * filter_height * filter_width * sizeof(float)));
		checkCudaErrors(cudaMalloc(&b, 1 * output_feature * 1 * 1 * sizeof(float)));
		checkCudaErrors(cudaMalloc(&workspace, workspace_size));
		checkCudaErrors(cudaMalloc(&pooling_layer_output, batch_size * input_feature * pooling_output_rows * pooling_output_cols * sizeof(float)));

		int n_iters = 100;
		cudaEvent_t start, stop;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		if (!conv_test) {
			std::vector<float> pool_compute_times;
			for (int i = 0; i < n_iters; i++) {
				float milli = 0;
				checkCudaErrors(cudaEventRecord(start));
				checkCUDNN(cudnnPoolingForward(cudnn_handle, pool_desc,
												&alpha,
												input_tensor, layer_input,
												&beta,
												pooling_output_tensor, pooling_layer_output));

				checkCudaErrors(cudaEventRecord(stop));
				checkCudaErrors(cudaEventSynchronize(stop));
				checkCudaErrors(cudaEventElapsedTime(&milli, start, stop));
				pool_compute_times.push_back(milli);
			}
			fstream f_pool_compute;
			f_pool_compute.open("pool_compute_time.dat", ios_base::out);
			for (int i = 0; i < n_iters; i++) {
				f_pool_compute << pool_compute_times[i] << endl;
			}
			f_pool_compute.close();
			exit(0);
		}

		std::vector<float> compute_times;
		for (int i = 0; i < n_iters; i++) {
			float milli = 0;
			checkCudaErrors(cudaEventRecord(start));

			checkCUDNN(cudnnConvolutionForward(cudnn_handle, &alpha, 
												input_tensor, layer_input,
												filter_desc, W,
												conv_desc, conv_fwd_algo,
												workspace, workspace_size,
												&beta,
												output_tensor, layer_output));
			checkCUDNN(cudnnAddTensor(cudnn_handle, &alpha, 
										bias_tensor, b, 
										&alpha,
										output_tensor, layer_output));

			checkCUDNN(cudnnActivationForward(cudnn_handle, actv_desc,
												&alpha,
												output_tensor, layer_output,
												&beta,
												output_tensor, layer_output));

			checkCudaErrors(cudaEventRecord(stop));
			checkCudaErrors(cudaEventSynchronize(stop));
			checkCudaErrors(cudaEventElapsedTime(&milli, start, stop));
			compute_times.push_back(milli);
		}

		void *h_layer_input;
		checkCudaErrors(cudaMallocHost(&h_layer_input, batch_size * input_feature * input_rows * input_cols * sizeof(float)));
		std::vector<float> transfer_times;
		for (int i = 0; i < n_iters; i++) {
			float milli;
			checkCudaErrors(cudaEventRecord(start));			
			checkCudaErrors(cudaMemcpyAsync(h_layer_input, layer_input, batch_size * input_feature * input_rows * input_cols * sizeof(float), cudaMemcpyDeviceToHost, NULL));
			checkCudaErrors(cudaEventRecord(stop));
			checkCudaErrors(cudaEventSynchronize(stop));
			checkCudaErrors(cudaEventElapsedTime(&milli, start, stop));
			transfer_times.push_back(milli);
		}

		fstream f_compute;
		fstream f_transfer;
		char filter_char[10];
		sprintf(filter_char, "%d", filter_size);
		std::string compute_filename = "compute_time_";
		std::string transfer_filename = "transfer_time_";
		compute_filename.append(filter_char);
		compute_filename.append(".dat");
		transfer_filename.append(filter_char);
		transfer_filename.append(".dat");
		f_compute.open(compute_filename.c_str(), ios_base::out);
		f_transfer.open(transfer_filename.c_str(), ios_base::out);
		for (int i = 0; i < n_iters; i++) {
			f_compute << compute_times[i] << std::endl;
			f_transfer << transfer_times[i] << std::endl;
		}
		f_compute.close();
		f_transfer.close();

		exit(0);
		

	}

	~Context() {
		checkCUBLAS(cublasDestroy(cublasHandle));
		checkCURAND(curandDestroyGenerator(curandgen));

	}

	void forwardPropagate(bool train=true) {
		
		

		

	}

	

	void train(int num_iter, float *train_images, float *train_labels, float *test_images, float *test_labels, int N) {
		// int image_size = rows * cols * channels;

		for (int iter = 0; iter < num_iter; iter++) {
			int image_id = iter % (N / batch_size);


			this->forwardPropagate();
			

			checkCudaErrors(cudaDeviceSynchronize());
			exit(0);
			
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
			// checkCudaErrors(cudaMemcpy(X, &test_images[start * input_size], input_size * size * sizeof(float), cudaMemcpyHostToDevice));
			// checkCudaErrors(cudaMemcpy(y, &test_labels[start], size * sizeof(float), cudaMemcpyHostToDevice));
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

int main(int argc, char *argv[]) {
	float *f_train_images, *f_train_labels, *f_test_images, *f_test_labels;
	int input_size = rows * cols * channels;
	f_train_images = (float *)malloc(N_train * input_size * sizeof(float));
	f_train_labels = (float *)malloc(N_train * sizeof(float));
	f_test_images = (float *)malloc(N_test * input_size * sizeof(float));
	f_test_labels = (float *)malloc(N_test * sizeof(float));
	

	float l_rate = 1e-3;
	int hidden_size = 50;
	int batch_size = 16;
	int output_size = 10;
	int filter_size = 3;
	if (argc >= 2) {
		filter_size = atoi(argv[1]);
	}
	Context context(input_size, batch_size, hidden_size, l_rate, output_size, filter_size);
	int n_iter = 10000;
	int n_rep = 10;


	for (int i = 0; i < n_rep; i++) {
		context.train(n_iter, f_train_images, f_train_labels, f_test_images, f_test_labels, N_train);
		cout << context.test(f_test_images, f_test_labels, N_test) << endl;
	}


}