#include <cuda.h>
#include <iostream>

__global__
void addKernel(int* A_d, int* B_d, int*C_d); //vector addition(device code)
void vecAdd(int* A, int* B, int* C, int n); //loading, transfer, execution(host code)

int main(void){
    const int SIZE=5;
    int a[SIZE]={1,2,3,4,5};
    int b[SIZE]={10,20,30,40,50};
    int c[SIZE]={0};

    vecAdd(a,b,c,5);
    
}

// Compute vector sum C = A+B
// Each thread performs one pairwise addition
__global__
void addKernel(int* A_d, int* B_d, int*C_d)
{
    // each thread knows its own index
    int i = threadIdx.x;
    C_d[i] = A_d[i] + B_d[i];
}

void vecAdd(int* A, int* B, int* C, int n)
{
    int size = n * sizeof(int);
    int* A_d=0;
    int* B_d=0;
    int* C_d=0;
    // Allocate device memory
    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);
    // Transfer A and B to device memory
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    
    //configure grid ==> <<<number of thread blocks within grid, number of threads in each thread block>>> 
    addKernel<<<1, size>>>(A_d, B_d, C_d);
    // Transfer C from device to host
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    // Free device memory for A, B, C
    cudaFree(A_d); cudaFree(B_d); cudaFree (C_d);
}