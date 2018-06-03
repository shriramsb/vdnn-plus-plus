#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
#include "include/cnmem.h"
#define NUM 100
__global__ void test_kernel(int *input,int num_requests) 
{
  int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  
  if(thread_id<num_requests)
  {
    input[thread_id]=input[thread_id]+thread_id;    
  }
}


int main(int argc,char **argv)
{
  int *device_data,*host_data;
  FILE *stats;
  stats=fopen("output_stats.txt","w");
  unsigned int num_blocks;
  int i=0;
  cnmemDevice_t device;
  memset(&device, 0, sizeof(device));
  //device.size = 2048;
  int *device_array[10];
 
  cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);
  host_data=(int*)malloc(NUM*sizeof(int));
  //cudaMalloc((void**)&device_data, sizeof(int)*NUM);
  cnmemPrintMemoryState(stats, NULL);
  cnmemMalloc((void**)&device_data, sizeof(int)*NUM,NULL);
  for(i=0;i<NUM;i++)
  {
    host_data[i]=i;  
  }

  size_t free,total;
  cnmemMemGetInfo(&free, &total, NULL);
  cudaMemcpy(device_data,host_data,NUM*sizeof(int),cudaMemcpyHostToDevice);
  num_blocks=ceil((double)NUM/(double)512);
  printf("Num Blocks = %u\n",num_blocks);
  test_kernel<<<num_blocks,512>>>(device_data,NUM);
  cudaMemcpy(host_data,device_data,NUM*sizeof(int),cudaMemcpyDeviceToHost);
 /*
  for(i=0;i<NUM;i++)
  {
    printf("Val=%d\n",host_data[i]);
  }*/
  if(stats==NULL)
    printf("FILE!!!ISSUE!!\n");
  printf("Free mem=%d, Total Mem=%d\n",(int)free,(int)total);
  cnmemPrintMemoryState(stats, NULL);
  //fprintf(stats,"Hello");
  cnmemFree(device_data,NULL);
  cnmemPrintMemoryState(stats, NULL);
  
  //Allocate device_array[i] one by one and print memory states
  
  cnmemMalloc((void**)&device_array[0], sizeof(int)*NUM,NULL);
  cnmemPrintMemoryState(stats, NULL);
  cnmemMalloc((void**)&device_array[1], sizeof(int)*NUM,NULL);
  cnmemPrintMemoryState(stats, NULL);
  cnmemMalloc((void**)&device_array[2], sizeof(int)*NUM,NULL);
  cnmemPrintMemoryState(stats, NULL);
  cnmemMalloc((void**)&device_array[3], sizeof(int)*NUM,NULL);
  cnmemPrintMemoryState(stats, NULL);
  cnmemMalloc((void**)&device_array[4], sizeof(int)*NUM,NULL);
  cnmemPrintMemoryState(stats, NULL);
  
  //Deallocate device_array[i] one by one and print memory states

  cnmemFree(device_array[0],NULL);
  cnmemPrintMemoryState(stats, NULL);
  cnmemFree(device_array[1],NULL);
  cnmemPrintMemoryState(stats, NULL);

  cnmemFree(device_array[2],NULL);
  cnmemPrintMemoryState(stats, NULL);

  cnmemFree(device_array[3],NULL);
  cnmemPrintMemoryState(stats, NULL);

  cnmemMalloc((void**)&device_array[5], sizeof(int)*512,NULL);
  cnmemPrintMemoryState(stats, NULL);
  cnmemFinalize();
  std::free(host_data);
  return 0;
}
