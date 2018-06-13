#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
#include "cnmem.h"

using namespace std;

#define CNMEM_GRANULARITY 512

int main(int argc,char **argv)
{
	// int *device_data,*host_data;
	FILE *stats;
	stats=fopen("output_stats.txt","w");
	// size_t size1 = 716973056;
	cnmemDevice_t cnmem_device;
	cnmem_device.device = 0;
	cnmem_device.size = 9 * CNMEM_GRANULARITY;
	cnmem_device.numStreams = 0;
	cnmem_device.streams = NULL;
	cnmem_device.streamSizes = NULL;

	cnmemInit(1, &cnmem_device, CNMEM_FLAGS_CANNOT_GROW);

	void *p[5];
	cnmemMalloc(&p[0], 5 * CNMEM_GRANULARITY, NULL);
	cnmemPrintMemoryStateTogether(stats, NULL);
	cnmemMalloc(&p[1], 2 * CNMEM_GRANULARITY, NULL);
	cnmemPrintMemoryStateTogether(stats, NULL);
	cnmemMalloc(&p[2], CNMEM_GRANULARITY, NULL);
	cnmemPrintMemoryStateTogether(stats, NULL);
	cnmemFree(p[1], NULL);
	cnmemPrintMemoryStateTogether(stats, NULL);
	cnmemMalloc(&p[3], CNMEM_GRANULARITY, NULL);
	cnmemPrintMemoryStateTogether(stats, NULL);
	cnmemFree(p[0], NULL);
	cnmemPrintMemoryStateTogether(stats, NULL);
	cnmemFree(p[2], NULL);
	cnmemPrintMemoryStateTogether(stats, NULL);
	if (cnmemMalloc(&p[4], 8 * CNMEM_GRANULARITY, NULL) != CNMEM_STATUS_SUCCESS) {
		printf("bad size\n");
	}
	cnmemPrintMemoryStateTogether(stats, NULL);

	// cnmemMalloc()

 
	
 
	// size_t size2 = 303063040;
	// size_t size3 = 512;
	// void *p, *q, *r;
	// cnmemMalloc(&p, size1, NULL);
	// cnmemPrintMemoryStateTogether(stats, NULL);
	// cnmemMalloc(&q, size2, NULL);
	// cnmemPrintMemoryStateTogether(stats, NULL);
	// cnmemMalloc(&r, size3, NULL);
	// cnmemPrintMemoryStateTogether(stats, NULL);


}
