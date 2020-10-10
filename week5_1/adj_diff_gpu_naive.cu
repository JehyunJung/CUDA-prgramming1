#include <stdio.h>
#include <stdlib.h> // for rand(), malloc(), free()
#include <fcntl.h> // for open(), write()
#include <sys/stat.h>
#include "common.h"
#include <sys/time.h>
#define GRIDSIZE (8 * 1024)
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE * BLOCKSIZE) // 32M byte needed!
void genData(float* ptr, unsigned int size) {
    while (size--) {
        *ptr++ = (float)(rand() % 1000) / 1000.0F;
    }
}
// compute result[i] = input[i] – input[i-1]
__global__ void adj_diff_naive(float* g_result, float* g_input) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
     // each thread reads one element to s_data
    if(i>0){
        g_result[i] = g_input[i]-g_input[i-1];
    }
}
int main(void) {
    float* pSource = NULL;
    float* pResult = NULL;
    int i;
    struct timeval start_time, end_time;
    // malloc memories on the host-side
    pSource = (float*)malloc(TOTALSIZE * sizeof(float));
    pResult = (float*)malloc(TOTALSIZE * sizeof(float));
    // generate source data
    genData(pSource, TOTALSIZE);
    // CUDA: allocate device memory
    float* pSourceDev = NULL;
    float* pResultDev = NULL;
    CUDA_CHECK( cudaMalloc((void**)&pSourceDev, TOTALSIZE * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&pResultDev, TOTALSIZE * sizeof(float)) );
    // CUDA: copy from host to device
    CUDA_CHECK( cudaMemcpy(pSourceDev, pSource, TOTALSIZE * sizeof(float), cudaMemcpyHostToDevice) );
    // get current time
    cudaThreadSynchronize();
    gettimeofday(&start_time, NULL);
    // CUDA: launch the kernel: result[i] = input[i] - input[i-1]
    dim3 dimGrid(GRIDSIZE, 1, 1);
    dim3 dimBlock(BLOCKSIZE, 1, 1);
    adj_diff_naive<<<dimGrid, dimBlock>>>(pResultDev, pSourceDev);
    // get end time
    cudaThreadSynchronize();
    gettimeofday(&end_time, NULL);
    double operating_time = (double)(end_time.tv_sec)+(double)(end_time.tv_usec)/1000000.0 -
    ((double)(start_time.tv_sec)+(double)(start_time.tv_usec)/1000000.0);
    printf("Elapsed: %f seconds\n", (double)operating_time);
    // CUDA: copy from device to host
    CUDA_CHECK( cudaMemcpy(pResult, pResultDev, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost) );
    // print sample cases
    i = 1;
    printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
    i = TOTALSIZE - 1;
    printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
    i = TOTALSIZE / 2;
    printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
    // CUDA: free the memory
    CUDA_CHECK( cudaFree(pSourceDev) );
    CUDA_CHECK( cudaFree(pResultDev) );
    // free the memory
    free(pSource);
    free(pResult);
}    