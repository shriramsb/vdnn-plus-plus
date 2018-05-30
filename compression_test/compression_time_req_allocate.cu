#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include <vector>
#define COMPRESSION_BATCH_SIZE 32

using namespace std;

struct ThreadArg {
	float *original_data;
	long num_elements;
	int thread_num;
	float ***compressed_data;
	bool **compressed_data_taken;
	unsigned int *mask;
};

struct CompressedPos {
	long compressed_data_batch;
	long offset;
};

int n_threads = 8;
int n_compressed_data_batches = 8;

long layer_sizes_alexnet[] = {56l * 56 * 96, 28l * 28 * 96, 27l * 27 * 256, 13l * 13 * 256, 13l * 12 * 384, 13l * 12 * 384, 13l * 13 * 256, 6l * 6 * 256};
bool layer_compress_alexnet[] = {true, false, true, true, true, false, false, false};
long layer_density_alexnet[] = {50, 80, 40, 60, 70, 70, 30, 60};
int num_layers_alexnet = 8;

long layer_sizes_vgg[] = {224l * 224 * 64, 
							224l * 224 * 64, 
							112l * 112 * 64, 
							112l * 112 * 128, 
							112l * 112 * 128, 
							56l * 56 * 128, 
							56l * 56 * 256, 
							56l * 56 * 256, 
							56l * 56 * 256, 
							28l * 28 * 256, 
							28l * 28 * 512, 
							28l * 28 * 512, 
							28l * 28 * 512, 
							14l * 14 * 512, 
							14l * 14 * 512, 
							14l * 14 * 512, 
							14l * 14 * 512, 
							7l * 7 * 512};

long layer_density_vgg[] = {50,
							20, 
							30,
							20,
							10,
							20,
							20,
							20,
							10,
							20,
							20,
							10,
							10,
							10,
							20,
							20,
							10,
							15
							};
bool layer_compress_vgg[] = {false,
						true,
						true,
						false,
						true,
						true,
						true,
						false,
						true,
						true,
						true,
						false,
						true,
						true,
						true,
						false,
						false,
						false};

int num_layers_vgg = 18;



// long *layer_sizes = layer_sizes_alexnet;
// bool *layer_compress = layer_compress_alexnet;
// long *layer_density = layer_density_alexnet;
// int num_layers = num_layers_alexnet;


long *layer_sizes = layer_sizes_vgg;
bool *layer_compress = layer_compress_vgg;
long *layer_density = layer_density_vgg;
int num_layers = num_layers_vgg;


void *compressThread(void *arg) {
	ThreadArg *thread_arg = (ThreadArg *)arg;
	float *original_data = thread_arg->original_data;
	float ***compressed_data = thread_arg->compressed_data;
	bool **compressed_data_taken = thread_arg->compressed_data_taken;
	unsigned int *mask = thread_arg->mask;
	int thread_num = thread_arg->thread_num;
	long num_elements = thread_arg->num_elements;

	long start = thread_num * num_elements / n_threads;
	long n_compression_batches = num_elements / n_threads / COMPRESSION_BATCH_SIZE;

	long compressed_data_batch_size = num_elements / n_threads / n_compressed_data_batches;
	cudaMallocHost((void **)&compressed_data[thread_num], n_compressed_data_batches * sizeof(float *));
	cudaMallocHost((void **)&compressed_data_taken[thread_num], n_compressed_data_batches * sizeof(bool));
	for (int i = 0; i < n_compressed_data_batches; i++) {
		compressed_data_taken[thread_num][i] = false;
	}
	CompressedPos current_pos;
	current_pos.compressed_data_batch = -1, current_pos.offset = compressed_data_batch_size;
	for (long i = 0; i < n_compression_batches; i++) {
		long mask_pos = (i * COMPRESSION_BATCH_SIZE + start) / COMPRESSION_BATCH_SIZE;
		mask[mask_pos] = 0;
		for (long j = i * COMPRESSION_BATCH_SIZE + start; j < (i + 1) * COMPRESSION_BATCH_SIZE + start; j++) {
			if (original_data[j] > 0) {
				if (current_pos.offset == compressed_data_batch_size) {
					cudaMallocHost((void **)&compressed_data[thread_num][current_pos.compressed_data_batch + 1], compressed_data_batch_size * sizeof(float));
					compressed_data_taken[thread_num][current_pos.compressed_data_batch + 1] = true;
					current_pos.compressed_data_batch = current_pos.compressed_data_batch + 1;
					current_pos.offset = 0;
				}
				mask[mask_pos] = (mask[mask_pos] << 1) + 1;
				compressed_data[thread_num][current_pos.compressed_data_batch][current_pos.offset] = original_data[j];
				current_pos.offset += 1;
			}
			else {
				mask[mask_pos] = (mask[mask_pos] << 1);
			}
		}
	}

	return NULL;
}

void *decompressThread(void *arg) {
	ThreadArg *thread_arg = (ThreadArg *)arg;
	float *original_data = thread_arg->original_data;
	float ***compressed_data = thread_arg->compressed_data;
	bool **compressed_data_taken = thread_arg->compressed_data_taken;
	unsigned int *mask = thread_arg->mask;
	int thread_num = thread_arg->thread_num;
	long num_elements = thread_arg->num_elements;

	long start = thread_num * num_elements / n_threads;
	long n_compression_batches = num_elements / n_threads / COMPRESSION_BATCH_SIZE;

	long compressed_data_batch_size = num_elements / n_threads / n_compressed_data_batches;
	// cudaMallocHost((void **)&compressed_data[thread_num], n_compressed_data_batches * sizeof(float *));
	CompressedPos current_pos;
	current_pos.compressed_data_batch = 0, current_pos.offset = 0;
	for (long i = 0; i < n_compression_batches; i++) {
		long mask_pos = (i * COMPRESSION_BATCH_SIZE + start) / COMPRESSION_BATCH_SIZE;
		for (long j = i * COMPRESSION_BATCH_SIZE + start; j < (i + 1) * COMPRESSION_BATCH_SIZE + start; j++) {
			if (mask[mask_pos] & 0x80000000 > 0) {
				original_data[j] = compressed_data[thread_num][current_pos.compressed_data_batch][current_pos.offset];
				current_pos.offset += 1;
				if (current_pos.offset == compressed_data_batch_size) {
					current_pos.compressed_data_batch += 1;
					current_pos.offset = 0;
				}
			}
			else {
				original_data[j] = 0;
			}
			mask[mask_pos] = mask[mask_pos] << 1;
		}
	}
	for (int i = 0; i < n_compressed_data_batches; i++) {
		if (compressed_data_taken[thread_num][i])
			cudaFreeHost(compressed_data[thread_num][i]);
		else
			break;
	}
	cudaFreeHost(compressed_data_taken[thread_num]);
	cudaFreeHost(compressed_data[thread_num]);
	return NULL;
}


int main() {
	int batch_size = 16;
	long total_space = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	vector<float ***> compressed_data_vec;
	vector<unsigned int *> mask_vec;
	vector<bool **> compressed_data_taken_vec; 

	pthread_t threads[n_threads];
	for (int i = 0; i < num_layers; i++) {
		layer_sizes[i] *= batch_size;
	}
	vector<float> compression_times;
	float total_milli = 0.0;
	for (int j = 0; j < num_layers; j++) {
		if (!layer_compress[j])
			continue;
		long num_elements = layer_sizes[j];
		float *original_data, ***compressed_data;
		bool **compressed_data_taken;
		unsigned int *mask;
		cudaMallocHost((void **)&original_data, num_elements * sizeof(float));
		// cudaMallocHost((void **)&compressed_data, num_elements * sizeof(float));

		// generate data
		for (long i = 0; i < num_elements; i++) {
			if (rand() % 100 < layer_density[j])
				original_data[i] = 1;
			else
				original_data[i] = 0;
		}
		if (num_elements % n_threads != 0) {
			cout << "bad number of threads" << endl;
			exit(0);
		}
		if ((num_elements / n_threads) % COMPRESSION_BATCH_SIZE != 0) {
			cout << "bad num_elements or n_threads" << endl;
			exit(0);
		}
		cout << "starting " << j << endl;
		cudaEventRecord(start);

		cudaMallocHost((void **)&compressed_data, n_threads * sizeof(float **));
		cudaMallocHost((void **)&mask, num_elements / COMPRESSION_BATCH_SIZE * sizeof(unsigned int));
		cudaMallocHost((void **)&compressed_data_taken, n_threads * sizeof(bool *));
		ThreadArg thread_arg[n_threads];
		for (int i = 0; i < n_threads; i++) {
			thread_arg[i].original_data = original_data;
			thread_arg[i].compressed_data = compressed_data;
			thread_arg[i].compressed_data_taken = compressed_data_taken;
			thread_arg[i].mask = mask;
			thread_arg[i].thread_num = i;
			thread_arg[i].num_elements = num_elements;
		}

		for (int i = 0; i < n_threads; i++) {
			pthread_create(&threads[i], NULL, &compressThread, (void *)&thread_arg[i]);
		}
		for (int i = 0; i < n_threads; i++) {
			pthread_join(threads[i], NULL);
		}
		compressed_data_vec.push_back(compressed_data);
		mask_vec.push_back(mask);
		compressed_data_taken_vec.push_back(compressed_data_taken);
		cudaFreeHost(original_data);
		// for (int i = 0; i < 27 * 27 * 256 * 128; i++);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milli;
		cudaEventElapsedTime(&milli, start, stop);
		compression_times.push_back(milli);
		total_milli += milli;
		// cout << milli << endl;
		cudaFreeHost(original_data);
		// cudaFreeHost(compressed_data);
		// cudaFreeHost(mask);
	}

	for (int i = 0; i < compression_times.size(); i++) {
		cout << compression_times[i] << endl;
	}
	cout << total_milli << endl;
	
	// calculating space consumed
	
	int k = 0;
	for (int j = 0; j < num_layers; j++) {
		long num_elements = layer_sizes[j];
		long cur_space = 0;
		if (!layer_compress[j]) {
			cur_space = num_elements * sizeof(float);
			total_space += cur_space;
			continue;
		}
		bool **compressed_data_taken = compressed_data_taken_vec[k];
		long compressed_data_batch_size = num_elements / n_threads / n_compressed_data_batches;
		for (int thread_num = 0; thread_num < n_threads; thread_num++) {
			for (int i = 0; i < n_compressed_data_batches; i++) {
				if (compressed_data_taken[thread_num][i])
					cur_space += compressed_data_batch_size;
				else
					break;
			}
		}
		// add size of mask
		cur_space += num_elements / COMPRESSION_BATCH_SIZE;
		cur_space *= sizeof(float);
		total_space += cur_space;
		k++;
	}
	cout << "total_space_compressed(MB): " << total_space * 1.0 / (1024 * 1024) << endl;

	// {
	// 	int n;
	// 	cout << "waiting..\n";
	// 	cin >> n;
	// }

	// decompression
	cout << "decompress" << endl;
	vector<float> decompression_times;
	float total_milli_decompress = 0.0;
	for (int j = num_layers - 1; j >= 0; j--) {
		if (!layer_compress[j])
			continue;
		long num_elements = layer_sizes[j];
		float *original_data, ***compressed_data;
		bool **compressed_data_taken;
		unsigned int *mask;
		compressed_data = compressed_data_vec.back();
		mask = mask_vec.back();
		compressed_data_taken = compressed_data_taken_vec.back();
		compressed_data_vec.pop_back();
		mask_vec.pop_back();
		compressed_data_taken_vec.pop_back();
		// cudaMallocHost((void **)&compressed_data, num_elements * sizeof(float));

		cout << "starting " << j << endl;
		cudaEventRecord(start);
		cudaMallocHost((void **)&original_data, num_elements * sizeof(float));
		ThreadArg thread_arg[n_threads];
		for (int i = 0; i < n_threads; i++) {
			thread_arg[i].original_data = original_data;
			thread_arg[i].compressed_data = compressed_data;
			thread_arg[i].compressed_data_taken = compressed_data_taken;
			thread_arg[i].mask = mask;
			thread_arg[i].thread_num = i;
			thread_arg[i].num_elements = num_elements;
		}

		for (int i = 0; i < n_threads; i++) {
			pthread_create(&threads[i], NULL, &decompressThread, (void *)&thread_arg[i]);
		}
		for (int i = 0; i < n_threads; i++) {
			pthread_join(threads[i], NULL);
		}
		cudaFreeHost(compressed_data_taken);
		cudaFreeHost(compressed_data);
		cudaFreeHost(mask);
		// cudaFreeHost(original_data);
		// for (int i = 0; i < 27 * 27 * 256 * 128; i++);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milli;
		cudaEventElapsedTime(&milli, start, stop);
		decompression_times.push_back(milli);
		total_milli_decompress += milli;
		// cout << milli << endl;
		// cudaFreeHost(compressed_data);
		// cudaFreeHost(mask);
	}

	for (int i = 0; i < decompression_times.size(); i++) {
		cout << decompression_times[i] << endl;
	}
	cout << total_milli_decompress << endl;

	// calculating total space
	total_space = 0;
	for (int j = 0; j < num_layers; j++) {
		long num_elements = layer_sizes[j];
		long cur_space = 0;
		cur_space = num_elements * sizeof(float);
		total_space += cur_space;
	}
	cout << "total space(MB): " << total_space * 1.0 / (1024 * 1024) << endl;


}