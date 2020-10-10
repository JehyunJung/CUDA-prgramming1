#include <cstdio>
#include <stdlib.h> // for rand(), malloc(), free()
#include "common.h"
#include <sys/time.h>
//CUDA kernel size settings
const int WIDTH = 1024; // total width is 1024*1024
const int TILE_WIDTH = 32; // block will be (TILE_WIDTH,TILE_WIDTH)
const int GRID_WIDTH = (WIDTH / TILE_WIDTH); // grid will be (GRID_WDITH,GRID_WDITH)
//random data generation
void genData(float* ptr, unsigned int size) {
    while (size--) {
        *ptr++ = (float)(rand() % 1000) / 1000.0F;
    }
}
__global__ void matmul(float* g_C, const float* g_A, const float* g_B, const int width) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
    int by = blockIdx.y; int bx = blockIdx.x;
    int ty = threadIdx.y; int tx = threadIdx.x;
    int gy = by * TILE_WIDTH + ty; // global y index
    int gx = bx * TILE_WIDTH + tx; // global x index
    float sum = 0.0F;
    for (register int m = 0; m < width / TILE_WIDTH; ++m) {
        // read into the shared memory blocks
        s_A[ty][tx] = g_A[gy * width + (m * TILE_WIDTH + tx)];
        s_B[ty][tx] = g_B[(m * TILE_WIDTH + ty) * width + gx];
        __syncthreads();
        // use the shared memory blocks to get the partial sum
        for (register int k = 0; k < TILE_WIDTH; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }
    g_C[gy * width + gx] = sum;
}
int main(void) {
    float* pA = NULL;
    float* pB = NULL;
    float* pC = NULL;
    struct timeval start_time, end_time;
    // malloc memories on the host-side
    pA = (float*)malloc(WIDTH * WIDTH * sizeof(float));
    pB = (float*)malloc(WIDTH * WIDTH * sizeof(float));
    pC = (float*)malloc(WIDTH * WIDTH * sizeof(float));
    // generate source data
    genData(pA, WIDTH * WIDTH);
    genData(pB, WIDTH * WIDTH);
    // CUDA: allocate device memory
    float* pAdev = NULL;
    float* pBdev = NULL;
    float* pCdev = NULL;
    CUDA_CHECK( cudaMalloc((void**)&pAdev, WIDTH * WIDTH * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&pBdev, WIDTH * WIDTH * sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&pCdev, WIDTH * WIDTH * sizeof(float)) );
    // copy from host to device
    CUDA_CHECK( cudaMemcpy(pAdev, pA, WIDTH * WIDTH * sizeof(float),
    cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(pBdev, pB, WIDTH * WIDTH * sizeof(float),
    cudaMemcpyHostToDevice) );
    //get current time
    cudaThreadSynchronize();
    gettimeofday(&start_time, NULL);
    // CUDA: launch the kernel
    dim3 dimGrid(GRID_WIDTH, GRID_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matmul <<< dimGrid, dimBlock>>>(pCdev, pAdev, pBdev, WIDTH);
    CUDA_CHECK( cudaPeekAtLastError() );
    //get current time
    cudaThreadSynchronize();
    gettimeofday(&end_time, NULL);
    double operating_time = (double)(end_time.tv_sec)+(double)(end_time.tv_usec)/1000000.0-
    ((double)(start_time.tv_sec)+(double)(start_time.tv_usec)/1000000.0);
    printf("Elapsed: %f seconds\n", (double)operating_time);
    // copy from device to host
    CUDA_CHECK( cudaMemcpy(pC, pCdev, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost) );
    // free device memory
    CUDA_CHECK( cudaFree(pAdev) );
    CUDA_CHECK( cudaFree(pBdev) );
    CUDA_CHECK( cudaFree(pCdev) );
    // print sample cases
    int i, j;
    i = 0; j = 0;
    printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
    i = WIDTH / 2; j = WIDTH / 2;
    printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
    i = WIDTH - 1; j = WIDTH - 1;
    printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
    // done
    return 0;
}