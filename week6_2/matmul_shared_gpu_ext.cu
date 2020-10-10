#include <cstdio>
#include <stdlib.h> // for rand(), malloc(), free()
#include "common.h"
#include <sys/time.h>
//CUDA kernel size settings
const int TILE_WIDTH = 32; // block will be (TILE_WIDTH,TILE_WIDTH)

void genData(float* ptr, unsigned int size);//random data generation
__global__ void matmulKernel(float * g_C, const float * g_A, const float * g_B, int HEIGHT,int COMMON, int WIDTH,int k);//matrix multiplication(device code) k-> ceil(COMMON/TILE_WIDTH)
void matrixMul(float * c,float * a,float * b, int HEIGHT,int COMMON, int WIDTH);//host->device data loading, transfer, execution
void printMatrix(float * matrix, int HEIGHT, int WIDTH); //print matrix 

int main(int argc, char * argv[]) {
    float* pA = NULL;
    float* pB = NULL;
    float* pC = NULL;
    int HEIGHT,COMMON, WIDTH;
    if(argc<4){
        printf("Not inserted Properly, Try again\n");
        printf("Ex: ./matmul_shared_gpu_ext <HEIGHT> <COMMON> <WIDTH>\n");
        exit(-1);
    }
    HEIGHT=atoi(argv[1]);
    COMMON=atoi(argv[2]);
    WIDTH=atoi(argv[3]);

    // malloc memories on the host-side
    pA = (float*)malloc(HEIGHT * COMMON * sizeof(float));
    pB = (float*)malloc(COMMON * WIDTH * sizeof(float));
    pC = (float*)malloc(HEIGHT * WIDTH * sizeof(float));

    // generate source data
    genData(pA, HEIGHT * COMMON);
    genData(pB, COMMON * WIDTH);
    matrixMul(pC, pA, pB, HEIGHT, COMMON, WIDTH);
    printMatrix(pC,HEIGHT,WIDTH);
    
    free(pA);
    free(pB);
    free(pC);

    return 0;
}

void genData(float* ptr, unsigned int size) {
    while (size) {
        *ptr++ = (float)size/(float)1000;
        size--;
     }
}

__global__ void matmulKernel(float* g_C, const float* g_A, const float* g_B, int HEIGHT,int COMMON, int WIDTH,int k) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
    int by = blockIdx.y; int bx = blockIdx.x;
    int ty = threadIdx.y; int tx = threadIdx.x;
    int gy = by * TILE_WIDTH + ty; // global y index
    int gx = bx * TILE_WIDTH + tx; // global x index
    float sum = 0.0F;
    for (register int m = 0; m < k; ++m) {
        // read into the shared memory blocks
        s_A[ty][tx] = g_A[gy * COMMON + (m * TILE_WIDTH + tx)];
        s_B[ty][tx] = g_B[(m * TILE_WIDTH + ty) * WIDTH + gx];
        
        __syncthreads();
        // use the shared memory blocks to get the partial sum
        for (register int k = 0; k < TILE_WIDTH; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }
    g_C[gy * WIDTH + gx] = sum;
}

void matrixMul(float * pC, float * pA, float * pB, int HEIGHT,int COMMON, int WIDTH){
    // CUDA: allocate device memory 
    float* pAdev = NULL;
    float* pBdev = NULL;
    float* pCdev = NULL;
    struct timeval start_time, end_time;

    CUDA_CHECK(cudaMalloc((void**)&pAdev, HEIGHT * COMMON * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&pBdev, COMMON * WIDTH * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&pCdev, HEIGHT * WIDTH * sizeof(float)));

    // copy from host to device
    CUDA_CHECK(cudaMemcpy(pAdev, pA, HEIGHT * COMMON * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pBdev, pB, COMMON * WIDTH * sizeof(float), cudaMemcpyHostToDevice));
    
    //get current time
    cudaThreadSynchronize();
    gettimeofday(&start_time, NULL);

    // CUDA: launch the kernel
    dim3 dimGrid(ceil((double)WIDTH/TILE_WIDTH),ceil((double)HEIGHT/TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matmulKernel <<< dimGrid, dimBlock>>>(pCdev, pAdev, pBdev, HEIGHT, COMMON, WIDTH,ceil((double)COMMON/TILE_WIDTH));
    //get current time
    cudaThreadSynchronize();
    gettimeofday(&end_time, NULL);
    double operating_time = (double)(end_time.tv_sec)+(double)(end_time.tv_usec)/1000000.0-((double)(start_time.tv_sec)+(double)(start_time.tv_usec)/1000000.0);
    printf("Elapsed: %f seconds\n", (double)operating_time);
    // copy from device to host

    CUDA_CHECK(cudaMemcpy(pC, pCdev, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost));
    // free device memory
    CUDA_CHECK(cudaFree(pAdev));
    CUDA_CHECK(cudaFree(pBdev));
    CUDA_CHECK(cudaFree(pCdev));
}

void printMatrix(float * matrix, int HEIGHT, int WIDTH){
    int i,j;
    for(i=0;i<HEIGHT;i++)
        for(j=0;j<WIDTH;j++)
            printf("c[%4d][%4d] = %f\n", i, j, matrix[i * WIDTH + j]);
}