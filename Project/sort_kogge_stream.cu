#include <stdio.h>
#include <stdlib.h>
#include<algorithm>
using namespace std;
#define BLOCKSIZE 256
#define DATASIZE 101
#define STREAMSIZE 10
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
	__shared__ int source[2*DATASIZE];
	__shared__ int destination[2*DATASIZE];
	__shared__ int dataCounter_s[DATASIZE];
	int tx=threadIdx.x;
	int temp;
	int stride=1;
	int index,i;

	dataCounter_s[tx]=0;
	source[tx]=0;
	destination[tx]=0;
	__syncthreads();

	for(i=0;i<STREAMSIZE;i++)
		dataCounter_s[tx]+=dataCounter_d[tx+i*DATASIZE];
	__syncthreads();
	source[DATASIZE+tx]=dataCounter_s[tx];
	__syncthreads();
	
	while(1){
		index=DATASIZE+tx;
		destination[index]=source[index]+source[index-stride];
		__syncthreads();
		stride*=2;
		if(stride>DATASIZE)
			break;
		
		//Swap between arrays
		temp=source[index];
		source[index]=destination[index];
		destination[index]=temp;
		__syncthreads();
	}

	for(i=destination[DATASIZE+tx-1];i<destination[DATASIZE+tx];i++){
		pResult_d[i]=tx;
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
	// start timer
	cudaEventRecord(start, 0);	

	pResult=(int*)malloc(input_size*sizeof(int));
	//Device Memory
	int *pSource_d;
	int *pResult_d;
	int *dataCounter_d;

	cudaMalloc((void**)&pSource_d,input_size*sizeof(int));
	cudaMalloc((void**)&pResult_d,input_size*sizeof(int));

	if(input_size>1000){
		//allocate host memory
		cudaMallocHost((void**)&pSource,input_size*sizeof(int));
		// generate source data
		genData(pSource, input_size);
		int segmentsize=input_size/STREAMSIZE;
		//Device memory allocation
		cudaMalloc((void**)&dataCounter_d,STREAMSIZE*DATASIZE*sizeof(int));
		cudaStream_t streams[STREAMSIZE];
		for(int i=0;i<STREAMSIZE;i++){
			cudaStreamCreate(&(streams[i]));
		}
		//Copy Host to Device
		
		for(int i=0;i<STREAMSIZE;i++){
			int offset=segmentsize*i;
			cudaMemcpyAsync(pSource_d+offset,pSource+offset,segmentsize*sizeof(int),cudaMemcpyHostToDevice,streams[i]);
		}
		for(int i=0;i<STREAMSIZE;i++){
			//launch kernel
			int offset=segmentsize*i;
			dim3 dimGrid(ceil((double)ceil((double)input_size/BLOCKSIZE)/segmentsize),1,1);
			dim3 dimBlock(ceil((double)segmentsize/BLOCKSIZE),1,1);
			countingData<<< dimGrid, dimBlock,0,streams[i]>>>(pSource_d+offset,dataCounter_d+DATASIZE*i,input_size);
		}
		for(int i=0;i<STREAMSIZE;i++){
			cudaStreamSynchronize(streams[i]);
			cudaStreamDestroy(streams[i]);
		}
		cudaFreeHost(pSource);
	}
	else{
			//allocate host memory
		pSource=(int*)malloc(input_size*sizeof(int));
		genData(pSource, input_size);
		cudaMalloc((void**)&dataCounter_d,DATASIZE*sizeof(int));
		//Copy Host to Device
		cudaMemcpy(pSource_d,pSource,input_size*sizeof(int),cudaMemcpyHostToDevice);
		//launch kernel
		dim3 dimGrid(ceil((double)input_size/BLOCKSIZE),1,1);
		dim3 dimBlock(BLOCKSIZE,1,1);
		countingData<<< dimGrid, dimBlock>>>(pSource_d,dataCounter_d,input_size);
		cudaDeviceSynchronize();
	}

	prefixSum<<<1,DATASIZE>>>(pResult_d,dataCounter_d);
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

