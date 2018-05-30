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
	float *compressed_data;
	unsigned int *mask;
};

int n_threads = 8;

long layer_sizes[] = {56l * 56 * 96, 28l * 28 * 96, 27l * 27 * 256, 13l * 13 * 256, 13l * 12 * 384, 13l * 12 * 384, 13l * 13 * 256, 6l * 6 * 256};
int num_layers = 8;

void *compressThread(void *arg) {
	ThreadArg *thread_arg = (ThreadArg *)arg;
	float *original_data = thread_arg->original_data;
	float *compressed_data = thread_arg->compressed_data;
	unsigned int *mask = thread_arg->mask;
	int thread_num = thread_arg->thread_num;
	long num_elements = thread_arg->num_elements;

	long start = thread_num * num_elements / n_threads;
	long n_compression_batches = num_elements / n_threads / COMPRESSION_BATCH_SIZE;

	for (long i = 0; i < n_compression_batches; i++) {
		long mask_pos = (i * COMPRESSION_BATCH_SIZE + start) / COMPRESSION_BATCH_SIZE;
		mask[mask_pos] = 0;
		for (long j = i * COMPRESSION_BATCH_SIZE + start; j < (i + 1) * COMPRESSION_BATCH_SIZE + start; j++) {
			if (original_data[j] > 0) {
				mask[mask_pos] = (mask[mask_pos] << 1) + 1;
				compressed_data[j] = original_data[j];
			}
			else {
				mask[mask_pos] = (mask[mask_pos] << 1);
			}

		}
	}

	return NULL;
}

int main() {
	int batch_size = 128;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	pthread_t threads[n_threads];
	for (int i = 0; i < num_layers; i++) {
		layer_sizes[i] *= batch_size;
	}
	vector<float> compression_times;
	float total_milli = 0.0;
	for (int j = 0; j < num_layers; j++) {
		long num_elements = layer_sizes[j];
		float *original_data, *compressed_data;
		unsigned int *mask;
		cudaMallocHost((void **)&original_data, num_elements * sizeof(float));
		// generate data
		for (long i = 0; i < num_elements; i++) {
			if (rand() % 10 < 3)
				original_data[i] = 0;
			else
				original_data[i] = 1;
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

		cudaMallocHost((void **)&compressed_data, num_elements * sizeof(float));
		cudaMallocHost((void **)&mask, num_elements / COMPRESSION_BATCH_SIZE * sizeof(unsigned int));

		ThreadArg thread_arg[n_threads];
		for (int i = 0; i < n_threads; i++) {
			thread_arg[i].original_data = original_data;
			thread_arg[i].compressed_data = compressed_data;
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
		// for (int i = 0; i < 27 * 27 * 256 * 128; i++);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milli;
		cudaEventElapsedTime(&milli, start, stop);
		compression_times.push_back(milli);
		total_milli += milli;
		// cout << milli << endl;
		cudaFreeHost(original_data);
		cudaFreeHost(compressed_data);
		cudaFreeHost(mask);
	}

	for (int i = 0; i < num_layers; i++) {
		cout << compression_times[i] << endl;
	}
	cout << total_milli << endl;
}