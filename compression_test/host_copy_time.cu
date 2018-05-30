#include <iostream>

using namespace std;

long layer_sizes[] = {56l * 56 * 96, 28l * 28 * 96, 27l * 27 * 256, 13l * 13 * 256, 13l * 12 * 384, 13l * 12 * 384, 13l * 13 * 256, 6l * 6 * 256};
int num_layers = 8;

int main() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	// long size_to_alloc = layer_sizes[0];
	
	for (int j = 0; j < num_layers; j++) {	
		long size_to_alloc = layer_sizes[j];
		int num_pieces = 8;
		void *p[num_pieces];
		for (int i = 0; i < num_pieces; i++) {
			cudaMallocHost(&p[i], size_to_alloc / num_pieces);
		}
		for (int i = 0; i < num_pieces; i++) {
			cudaFreeHost(p[i]);
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milli;
	cudaEventElapsedTime(&milli, start, stop);
	cout << "allocating and freeing pieces(ms): " << milli << endl;

	void *p_bulk;


	cudaEventRecord(start);
	for (int j = 0; j < num_layers; j++) {
		long size_to_alloc = layer_sizes[j];
		cudaMallocHost(&p_bulk, size_to_alloc);
		cudaFreeHost(p_bulk);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);
	cout << "allocating and freeing bulk(ms): " << milli << endl;	

}