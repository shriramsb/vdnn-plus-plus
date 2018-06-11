#include <iostream>
#include <cnmem.h>
#define CNMEM_GRANULARITY 512

using namespace std;

int main() {
	cnmemDevice_t cnmem_device;
	cnmem_device.device = 0;
	cnmem_device.size = CNMEM_GRANULARITY;
	cnmem_device.numStreams = 0;
	cnmem_device.streams = NULL;
	cnmem_device.streamSizes = NULL;
	cnmemInit(1, &cnmem_device, CNMEM_FLAGS_DEFAULT);

	void *p, *q;
	cout << "cnmem initialized\n";
	cnmemMalloc(&p, CNMEM_GRANULARITY, NULL);
	cnmemMalloc(&q, CNMEM_GRANULARITY, NULL);
	size_t free, total;
	cnmemMemGetInfo(&free, &total, NULL);
	cout << "free: " << free << endl;
	cout << "total: " << total << endl;

	cnmemFree(p, NULL);
	cnmemFree(q, NULL);
	cnmemMalloc(&p, CNMEM_GRANULARITY * 2, NULL);
	cnmemMemGetInfo(&free, &total, NULL);
	cout << "free: " << free << endl;
	cout << "total: " << total << endl;
	cnmemFinalize();

	cnmemInit(1, &cnmem_device, CNMEM_FLAGS_DEFAULT);
	
	cnmemMemGetInfo(&free, &total, NULL);
	cout << "free: " << free << endl;
	cout << "total: " << total << endl;	
	cnmemMalloc(&p, CNMEM_GRANULARITY, NULL);
	cnmemMemGetInfo(&free, &total, NULL);
	cout << "free: " << free << endl;
	cout << "total: " << total << endl;

	FILE *separate_stats;
	FILE *together_stats;
	separate_stats = fopen("separate_stats.dat", "w");
	together_stats = fopen("together_stats.dat", "w");
	cnmemPrintMemoryState(separate_stats, NULL);
	cnmemPrintMemoryStateTogether(together_stats, NULL);
	fclose(separate_stats);
	fclose(together_stats);

}