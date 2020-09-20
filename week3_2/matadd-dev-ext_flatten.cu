#include <cuda.h>
#include <iostream>
#include <time.h>
#include "common.h"
// Compute vector sum C = A+B
// Each thread performs one pairwise addition
__global__ void addKernel(const int* dev_a, const int* dev_b, int*dev_c, int n)
{
    int x = (blockIdx.x*blockDim.x)+threadIdx.x; //global index of x
    int y = (blockIdx.y*blockDim.y)+threadIdx.y; //global index of y
    int i = y * (blockDim.x) + x;   //actual index of i(1D array)
    //error handling
    if(i<n)
        dev_c[i] = dev_a[i] + dev_b[i];
}
int main(int argc, char*argv[]){
    int HEIGHT, WIDTH;
    int x,y;
    int *a,*b,*c;
    int *dev_a,*dev_b,*dev_c;
    int SIZE;
    
    if(argc<3){
        printf("Not inserted block size, Try again\n");
        exit(-1);
    }
    HEIGHT=atoi(argv[1]);
    WIDTH=atoi(argv[2]);
    SIZE=HEIGHT*WIDTH;

    a=(int*)malloc(sizeof(int*)*SIZE);
    b=(int*)malloc(sizeof(int*)*SIZE);
    c=(int*)malloc(sizeof(int*)*SIZE);


   for(y=0;y<HEIGHT;y++){
       for(x=0;x<WIDTH;x++){
           a[y*WIDTH+x]=y*10+x;
           b[y*WIDTH+x]=(y*10+x)*100;
        }
   }

    // Allocate device memory
    CUDA_CHECK( cudaMalloc((void**)&dev_a, SIZE*sizeof(int)) );
    CUDA_CHECK( cudaMalloc((void**)&dev_b, SIZE*sizeof(int)) );
    CUDA_CHECK( cudaMalloc((void**)&dev_c, SIZE*sizeof(int)) );

    // Transfer A and B to device memory
    CUDA_CHECK(cudaMemcpy(dev_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, b, SIZE*sizeof(int), cudaMemcpyHostToDevice));

    //dimension of thread block(x,y,z)
    dim3 dimBlock(WIDTH,HEIGHT,1);
    
    //configure grid ==> <<<number of thread blocks within grid, number of threads in each thread block>>> 
    addKernel<<<ceil(SIZE/256.0), 256>>>(dev_a, dev_b, dev_c,SIZE);
    
    // Transfer C from device to host
    CUDA_CHECK(cudaMemcpy(c, dev_c, SIZE*sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory for A, B, C
    CUDA_CHECK(cudaFree(dev_a)); 
    CUDA_CHECK(cudaFree(dev_b)); 
    CUDA_CHECK(cudaFree(dev_c));

    for(y=0;y<HEIGHT;y++){
        for(x=0;x<WIDTH;x++)
            printf("%5d",c[y*WIDTH+x]);  
        printf("\n");
    }

    free(a);
    free(b);
    free(c);
    return 0;
}
