#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include <vector>
#include <cmath>
#define NUM_COMPRESSION_THREADS 8
#define COMPRESSION_DISCRETIZATION_FACTOR 8
#define COMPRESSION_BATCH_SIZE 32
#define ALLOC_AND_COMPRESS 0
#define BYTE_SIZE 8

using namespace std;

struct CompressedData {
	void ***data;
	bool **slot_taken;
	unsigned int *mask;
};

struct CompressionMetadata {
	long total_compression_batches;
	long *num_elements, *start_pos, *num_compression_batches;
	long **slot_size; 
};

struct CompressionThreadArgs {
	CompressedData *compressed_data;
	float *original_data;
	CompressionMetadata *compression_metadata;
	int thread_num;
};

struct Position2dArray {
	long slot;
	long offset;
};

long layer_sizes_alexnet[] = {56l * 56 * 96, 28l * 28 * 96, 27l * 27 * 256, 13l * 13 * 256, 13l * 12 * 384, 13l * 12 * 384, 13l * 13 * 256, 6l * 6 * 256};
bool layer_compress_alexnet[] = {false, false, true, false, true, true, true, true};
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
bool layer_compress_vgg[] = {true,
						true,
						true,
						true,
						true,
						true,
						true,
						true,
						true,
						true,
						true,
						true,
						true,
						true,
						true,
						true,
						true,
						true};

int num_layers_vgg = 18;

long *layer_sizes = layer_sizes_alexnet;
bool *layer_compress = layer_compress_alexnet;
long *layer_density = layer_density_alexnet;
int num_layers = num_layers_alexnet;

void *compressThread(void *);
void *decompressThread(void *);

int main() {
	int batch_size = 128;
	for (int i = 0; i < num_layers; i++) {
		layer_sizes[i] *= batch_size;
	}
	// allocate space for compressed_data_pointers
	CompressedData *compressed_data = (CompressedData *)malloc(num_layers * sizeof(CompressedData));
	for (int i = 0; i < num_layers; i++) {
		if (layer_compress[i]) {
			cudaMallocHost((void **)&compressed_data[i].data, NUM_COMPRESSION_THREADS * sizeof(void **));
			cudaMallocHost((void **)&compressed_data[i].slot_taken, NUM_COMPRESSION_THREADS * sizeof(bool *));
			for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
				cudaMallocHost((void **)&compressed_data[i].data[j], COMPRESSION_DISCRETIZATION_FACTOR * sizeof(void *));
				cudaMallocHost((void **)&compressed_data[i].slot_taken[j], COMPRESSION_DISCRETIZATION_FACTOR * sizeof(bool));
			}
		}
	}

	if (COMPRESSION_BATCH_SIZE != sizeof(unsigned int) * BYTE_SIZE) {
		std::cout << "Panic!! COMPRESSION_BATCH_SIZE = 32\n sizeof(unsigned int) = " << sizeof(unsigned int) << endl;
		std::cout << "Sizes do not match\n";
	}

	CompressionMetadata *compression_metadata = (CompressionMetadata *)malloc(num_layers * sizeof(CompressionMetadata));
	for (int i = 0; i < num_layers; i++) {
		if (layer_compress[i]) {
			compression_metadata[i].num_compression_batches = (long *)malloc(NUM_COMPRESSION_THREADS * sizeof(long));
			compression_metadata[i].num_elements = (long *)malloc(NUM_COMPRESSION_THREADS * sizeof(long));
			compression_metadata[i].start_pos = (long *)malloc(NUM_COMPRESSION_THREADS * sizeof(long));
			compression_metadata[i].slot_size = (long **)malloc(NUM_COMPRESSION_THREADS * sizeof(long *));
			for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
				compression_metadata[i].slot_size[j] = (long *)malloc(COMPRESSION_DISCRETIZATION_FACTOR * sizeof(long));
			}
			compression_metadata[i].total_compression_batches = ceil(1.0 * layer_sizes[i] / COMPRESSION_BATCH_SIZE);
			for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
				compression_metadata[i].num_compression_batches[j] = compression_metadata[i].total_compression_batches / NUM_COMPRESSION_THREADS;
			}
			long num_leftout_compression_batches = compression_metadata[i].total_compression_batches % NUM_COMPRESSION_THREADS;
			for (int j = 0; j < num_leftout_compression_batches; j++) {
				compression_metadata[i].num_compression_batches[NUM_COMPRESSION_THREADS - 1 - j] += 1;
			}
			for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
				if (j < NUM_COMPRESSION_THREADS - 1) {
					compression_metadata[i].num_elements[j] = compression_metadata[i].num_compression_batches[j] * COMPRESSION_BATCH_SIZE;
				}
				else {
					compression_metadata[i].num_elements[j] = (compression_metadata[i].num_compression_batches[j] - 1) * COMPRESSION_BATCH_SIZE;
					if (layer_sizes[i] % COMPRESSION_BATCH_SIZE == 0)
						compression_metadata[i].num_elements[j] += COMPRESSION_BATCH_SIZE;
					else
						compression_metadata[i].num_elements[j] += layer_sizes[i] % COMPRESSION_BATCH_SIZE;
				}
			}
			long cumulative_start_pos = 0;
			for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
				compression_metadata[i].start_pos[j] = cumulative_start_pos;
				cumulative_start_pos += compression_metadata[i].num_elements[j];
			}
			for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
				for (int k = 0; k < COMPRESSION_DISCRETIZATION_FACTOR; k++) {
					compression_metadata[i].slot_size[j][k] = compression_metadata[i].num_elements[j] / COMPRESSION_DISCRETIZATION_FACTOR;
				}
				for (int k = 0; k < compression_metadata[i].num_elements[j] % COMPRESSION_DISCRETIZATION_FACTOR; k++) {
					compression_metadata[i].slot_size[j][k] += 1;
				}
			}
		}
	}

	float **h_layer_input = (float **)malloc(num_layers * sizeof(float *));
	// create args for compression threads
	CompressionThreadArgs **compression_thread_args = (CompressionThreadArgs **)malloc(num_layers * sizeof(CompressionThreadArgs *));
	for (int i = 0; i < num_layers; i++) {
		compression_thread_args[i] = (CompressionThreadArgs *)malloc(NUM_COMPRESSION_THREADS * sizeof(CompressionThreadArgs));
		for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
			compression_thread_args[i][j].compressed_data = &compressed_data[i];
			compression_thread_args[i][j].original_data = h_layer_input[i];
			compression_thread_args[i][j].compression_metadata = &compression_metadata[i];
			compression_thread_args[i][j].thread_num = j;
		}
	}
	

	vector<float> compression_times, decompression_times;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	for (int i = 0; i < num_layers; i++) {
		if (!layer_compress[i]) {
			compression_times.push_back(0);
			continue;
		}

		float milli;
		cudaMallocHost((void **)&h_layer_input[i], layer_sizes[i] * sizeof(float));
		pthread_t threads[NUM_COMPRESSION_THREADS];

		// generate data
		for (long j = 0; j < layer_sizes[i]; j++) {
			if (rand() % 100 < layer_density[i])
				h_layer_input[i][j] = 1;
			else
				h_layer_input[i][j] = 0;
		}

		cout << "starting " << i << endl;
		cudaEventRecord(start);

		cudaMallocHost(&compressed_data[i].mask, ceil(1.0 * layer_sizes[i] / COMPRESSION_BATCH_SIZE) * sizeof(unsigned int));
		for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
			compression_thread_args[i][j].original_data = h_layer_input[i];
		}

		for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
			pthread_create(&threads[j], NULL, &compressThread, (void *)&compression_thread_args[i][j]);
		}
		for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
			pthread_join(threads[j], NULL);
		}

		cudaFreeHost(h_layer_input[i]);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milli, start, stop);
		compression_times.push_back(milli);
	}

	size_t total_size = 0;
	size_t total_size_uncompressed = 0;
	for (int i = 0; i < num_layers; i++) {
		total_size_uncompressed += layer_sizes[i];
		if (!layer_compress[i]) {
			total_size += layer_sizes[i];
			continue;
		}
		for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
			for (int k = 0; k < COMPRESSION_DISCRETIZATION_FACTOR; k++) {
				if (compressed_data[i].slot_taken[j][k]) {
					total_size += compression_metadata[i].slot_size[j][k];
				}
			}
		}
	}
	std::cout << "total_size_compressed(MB): " << total_size * sizeof(float) / (1.0 * 1024 * 1024) << std::endl;
	std::cout << "total_size_uncompressed(MB): " << total_size_uncompressed * sizeof(float) / (1.0 * 1024 * 1024) << std::endl;


	// decompression
	for (int i = 0; i < num_layers; i++) {
		float milli;
		if (!layer_compress[i]) {
			decompression_times.push_back(0);
			continue;
		}

		pthread_t threads[NUM_COMPRESSION_THREADS];
		cudaEventRecord(start);
		cudaMallocHost(&h_layer_input[i], layer_sizes[i] * sizeof(float));

		cout << "starting " << i << endl;



		for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
			compression_thread_args[i][j].original_data = h_layer_input[i];
		}

		for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
			pthread_create(&threads[j], NULL, &decompressThread, (void *)&compression_thread_args[i][j]);
		}
		for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
			pthread_join(threads[j], NULL);
		}

		for (int j = 0; j < NUM_COMPRESSION_THREADS; j++) {
			for (int k = 0; k < COMPRESSION_DISCRETIZATION_FACTOR; k++) {
				if (compressed_data[i].slot_taken[j][k]) {
					cudaFreeHost(compressed_data[i].data[j][k]);
				}
			}
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milli, start, stop);
		decompression_times.push_back(milli);
	}

	float total_compression_time = 0, total_decompression_time = 0;
	for (int i = 0; i < num_layers; i++) {
		cout << i << " " << compression_times[i] << endl;
		total_compression_time += compression_times[i];
	}
	cout << endl;
	for (int i = 0; i < num_layers; i++) {
		cout << i << " " << decompression_times[i] << endl;
		total_decompression_time += decompression_times[i];
	}
	cout << "total compression time(ms): " << total_compression_time << endl;
	cout << "total decompression time(ms): " << total_decompression_time << endl;

}

void *compressThread(void *arg) {
	CompressionThreadArgs *args = (CompressionThreadArgs *)arg;
	
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
#ifdef ALLOC_AND_COMPRESS
				cudaMallocHost((void **)&compressed_data.data[thread_num][compressed_data_pos.slot], compression_metadata.slot_size[thread_num][compressed_data_pos.slot] * sizeof(float));
#endif
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

void *decompressThread(void *arg) {
	CompressionThreadArgs *args = (CompressionThreadArgs *)arg;
	
	// retrieve args
	float *original_data = (float *)(args->original_data);
	CompressedData compressed_data = *(args->compressed_data);
	CompressionMetadata compression_metadata = *(args->compression_metadata);
	int thread_num = args->thread_num;

	Position2dArray compressed_data_pos, mask_pos;
	compressed_data_pos.slot = 0, compressed_data_pos.offset = 0;
	mask_pos.slot = compression_metadata.start_pos[thread_num] / COMPRESSION_BATCH_SIZE;
	mask_pos.offset = 0;

	// handling the last part where it might not be completely filled
	if (thread_num == NUM_COMPRESSION_THREADS - 1) {
		if (compression_metadata.num_elements[thread_num] % COMPRESSION_BATCH_SIZE != 0) {
			compressed_data.mask[mask_pos.slot + compression_metadata.num_elements[thread_num] / COMPRESSION_BATCH_SIZE] = compressed_data.mask[mask_pos.slot + compression_metadata.num_elements[thread_num] / COMPRESSION_BATCH_SIZE] << (COMPRESSION_BATCH_SIZE - compression_metadata.num_elements[thread_num] % COMPRESSION_BATCH_SIZE);
		}
	}

	for (long i = compression_metadata.start_pos[thread_num]; i < compression_metadata.start_pos[thread_num] + compression_metadata.num_elements[thread_num]; i++) {

		if (compressed_data.mask[mask_pos.slot] & 0x80000000 > 0) {
			original_data[i] = ((float ***)compressed_data.data)[thread_num][compressed_data_pos.slot][compressed_data_pos.offset];
			compressed_data_pos.offset += 1;
			if (compressed_data_pos.offset == compression_metadata.slot_size[thread_num][compressed_data_pos.slot]) {
				compressed_data_pos.slot += 1;
				compressed_data_pos.offset = 0;
			}
		}
		else {
			original_data[i] = 0;
		}
		compressed_data.mask[mask_pos.slot] = (compressed_data.mask[mask_pos.slot] << 1);
		mask_pos.offset += 1;
		if (mask_pos.offset == COMPRESSION_BATCH_SIZE) {
			mask_pos.slot += 1;
			mask_pos.offset = 0;
		}

	}
}