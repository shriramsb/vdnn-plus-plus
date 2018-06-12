#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
#include "cnmem.h"

using namespace std;


int main(int argc,char **argv)
{
  int *device_data,*host_data;
  FILE *stats;
  stats=fopen("output_stats.txt","w");
  size_t size1 = 716973056;
  cnmemDevice_t cnmem_device;
  cnmem_device.device = 0;
  cnmem_device.size = size1;
  cnmem_device.numStreams = 0;
  cnmem_device.streams = NULL;
  cnmem_device.streamSizes = NULL;

  cnmemInit(1, &cnmem_device, 0);
 
  size_t size2 = 303063040;
  size_t size3 = 512;
  void *p, *q, *r;
  cnmemMalloc(&p, size1, NULL);
  cnmemPrintMemoryStateTogether(stats, NULL);
  cnmemMalloc(&q, size2, NULL);
  cnmemPrintMemoryStateTogether(stats, NULL);
  cnmemMalloc(&r, size3, NULL);
  cnmemPrintMemoryStateTogether(stats, NULL);

}
