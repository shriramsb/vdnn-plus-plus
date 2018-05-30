#include <iostream>

using namespace std;

int main() {
	cudaEvent_t start, stop, done_offload;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&done_offload);
	void *p, *q;
	long size = 1024l * 1024 * 200;
	cudaMalloc(&p, size);
	cudaMallocHost(&q, size);
	cout << "without split by event\n";
	int N = 100;
	cudaEventRecord(start);
	for (int i = 0; i < N; i++) {
		cudaMemcpyAsync(q, p, size, cudaMemcpyDeviceToHost, NULL);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milli;
	cudaEventElapsedTime(&milli, start, stop);
	cout << "Time(ms): " << milli << endl;

	cout << "with split by event\n";
	cudaEventRecord(start);
	for (int i = 0; i < N; i++) {
		cudaMemcpyAsync(q, p, size, cudaMemcpyDeviceToHost, NULL);
		cudaEventRecord(done_offload, NULL);
		cudaEventSynchronize(done_offload);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);
	cout << "Time(ms): " << milli << endl;

}