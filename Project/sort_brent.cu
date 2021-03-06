#include <stdio.h>
#include <stdlib.h>
#include<algorithm>
using namespace std;
#define BLOCKSIZE 256
#define DATASIZE 101

//INSERT CODE HERE---------------------------------
//Counting Sort
__global__ void countingData(int * pSource_d,int *dataCounter_d,int input_size){
	//Shared memory for saving data counts
	__shared__ int dataCounter_s[DATASIZE];

	int tx=threadIdx.x;
	int gx=blockIdx.x*blockDim.x + tx;
	
	//set initial value of array elements to 0
	if(tx<DATASIZE)
		dataCounter_s[tx]=0;
	__syncthreads();	
		
	if(gx<input_size)
		//atomically counts data
		atomicAdd(&(dataCounter_s[pSource_d[gx]]),1);
	__syncthreads();
	
	//add all shared memory values
	if(tx<DATASIZE)
		atomicAdd(&(dataCounter_d[tx]),dataCounter_s[tx]);
}

//Prefix Sum(Double-Buffered Kogge-Stone Parallel Scan Algorithm)
__global__ void prefixSum(int * pResult_d, int * dataCounter_d){
	__shared__ int T[DATASIZE];
	int stride=1;
	int index,i;
	int tx=threadIdx.x;
	
	T[tx]=dataCounter_d[tx];
	index=DATASIZE/2+tx;
	T[index]=dataCounter_d[index];
	__syncthreads();

	while(stride<DATASIZE){
		index=(tx+1)*stride*2-1;
		if(index<DATASIZE)
			T[index]+=T[index-stride];
		stride*=2;

		__syncthreads();
	}
	stride/=2;
	while(stride>=1){
		index=(tx+1)*stride*2-1;
		if(index<DATASIZE && (index+stride)<DATASIZE)
			T[index+stride]+=T[index];
		stride/=2;
		
		__syncthreads();
	}

	if(tx==0)
		for(i=0;i<T[tx];i++)
			pResult_d[i]=tx;
		
	else{
		index=tx;
		for(i=T[index-1];i<T[index];i++)
			pResult_d[i]=index;
		index=tx+DATASIZE/2;
		if(index<DATASIZE)
			for(i=T[index-1];i<T[index];i++)
				pResult_d[i]=index;	
	}
}

void verify(int* src, int*result, int input_size){
	sort(src, src+input_size);
	long long match_cnt=0;
	for(int i=0; i<input_size;i++)
	{
		if(src[i]==result[i])
			match_cnt++;
	}

	if(match_cnt==input_size)
		printf("TEST PASSED\n\n");
	else
		printf("TEST FAILED\n\n");

}

void genData(int* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (int)(rand() % 101);
	}
}

int main(int argc, char* argv[]) {
	int* pSource = NULL;
	int* pResult = NULL;
	int input_size=0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (argc == 2)
		input_size=atoi(argv[1]);
	else
	{
    		printf("\n    Invalid input parameters!"
	   		"\n    Usage: ./sort <input_size>"
           		"\n");
        	exit(0);
	}

	//allocate host memory
	pSource=(int*)malloc(input_size*sizeof(int));
	pResult=(int*)malloc(input_size*sizeof(int));
	// generate source data
	genData(pSource, input_size);
	
	
	// start timer
	cudaEventRecord(start, 0);

	//INSERT CODE HERE--------------------
	//Device Memory
	int *pSource_d;
	int *pResult_d;
	int *dataCounter_d;
	//Device memory allocation
	cudaMalloc((void**)&pSource_d,input_size*sizeof(int));
	cudaMalloc((void**)&pResult_d,input_size*sizeof(int));
	cudaMalloc((void**)&dataCounter_d,DATASIZE*sizeof(int));
	//Copy Host to Device
	cudaMemcpy(pSource_d,pSource,input_size*sizeof(int),cudaMemcpyHostToDevice);
	
	//launch kernel
	dim3 dimGrid(ceil((double)input_size/BLOCKSIZE),1,1);
    dim3 dimBlock(BLOCKSIZE,1,1);
    countingData<<< dimGrid, dimBlock>>>(pSource_d,dataCounter_d,input_size);
	cudaDeviceSynchronize();
	prefixSum<<<1,(DATASIZE/2)+1>>>(pResult_d,dataCounter_d);
	cudaDeviceSynchronize();
	//Copy Device to Host
	cudaMemcpy(pResult,pResult_d,input_size*sizeof(int),cudaMemcpyDeviceToHost);
	//Free Device Memory
	cudaFree(pSource_d);
	cudaFree(pResult_d);
	cudaFree(dataCounter_d);

	// end timer
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("elapsed time = %f msec\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("Verifying results..."); 
	fflush(stdout);
	verify(pSource, pResult, input_size);
	fflush(stdout);
}

