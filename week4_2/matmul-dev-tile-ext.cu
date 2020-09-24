#include <cuda.h>
#include <iostream>
#include "common.h"

__global__ void mulKernel(const int* dev_a, const int* dev_b, int* dev_c, int WIDTH); //matrix addition(device code)
void matrixMul(int** A, int** B, int** C, int WIDTH,int TILE_WIDTH); //loading, transfer, execution(host code)
void printArray(int **array, int WIDTH); // print Array elements
int main(int argc, char * argv[]){
    int WIDTH,TILE_WIDTH;
    int x,y;
    int **a,**b,**c;
    
    if(argc<2){
        printf("Not inserted block size,Tile size Try again\n");
        exit(-1);
    }
    //allocating program arguments
    WIDTH=atoi(argv[1]);
    TILE_WIDTH=atoi(argv[2]);

    //2D array dynamic allocation 
    a=(int**)malloc(sizeof(int*)*WIDTH);
    b=(int**)malloc(sizeof(int*)*WIDTH);
    c=(int**)malloc(sizeof(int*)*WIDTH);

    for(y=0;y<WIDTH;y++){
        a[y]=(int*)malloc(sizeof(int)*WIDTH);
        b[y]=(int*)malloc(sizeof(int)*WIDTH);
        c[y]=(int*)malloc(sizeof(int)*WIDTH);
    }
    
    //initializing 2D array
   for(y=0;y<WIDTH;y++){
       for(x=0;x<WIDTH;x++){
           a[y][x]=y*10+x;
           b[y][x]=(y*10+x)*100;
        }
   }


   clock_t start=clock();
   matrixMul(a,b,c,WIDTH,TILE_WIDTH);
   printArray(c,WIDTH);
   printf("Matrix Multiplication execution time(Parallel): %fms\n",(double)(clock()-start));

   //free 2D array
    for(y=0;y<WIDTH;y++){
        free(a[y]);
        free(b[y]);
        free(c[y]);
    }

    free(a);
    free(b);
    free(c);
    return 0;
}

// Compute matrix multiplication
// Each thread performs one pairwise addition
__global__ void mulKernel(const int* dev_a, const int* dev_b, int*dev_c, int WIDTH)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x; //global index of x
    int y = blockIdx.y*blockDim.y+threadIdx.y; //global index of y
    int i = y * WIDTH + x;   //actual index of i(1D array)
    int k,sum=0;
    
    for(k=0;k<WIDTH;k++)
        sum+=dev_a[y*WIDTH+k]*dev_b[k*WIDTH+x];
    dev_c[i]=sum;
}

void matrixMul(int** A, int** B, int** C, int WIDTH,int TILE_WIDTH)
{
    int* dev_a=0;
    int* dev_b=0;
    int* dev_c=0;
    int y=0;
    int SIZE=WIDTH*WIDTH;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&dev_a, SIZE*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_b, SIZE*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c, SIZE*sizeof(int)));

    // Transfer A and B to device memory (need flattening)
    for(y=0;y<WIDTH;y++){
        CUDA_CHECK(cudaMemcpy(&dev_a[y*WIDTH], A[y], WIDTH*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&dev_b[y*WIDTH], B[y], WIDTH*sizeof(int), cudaMemcpyHostToDevice));
    }
    
    dim3 dimGrid(ceil(WIDTH/TILE_WIDTH),ceil(WIDTH/TILE_WIDTH),1);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
    //configure grid ==> <<<number of thread blocks within grid, number of threads in each thread block>>> 
    mulKernel<<<dimGrid,dimBlock>>>(dev_a, dev_b, dev_c, WIDTH);

    // Transfer C from device to host
    for(y=0;y<WIDTH;y++){
        CUDA_CHECK(cudaMemcpy(C[y], &dev_c[y*WIDTH], WIDTH*sizeof(int), cudaMemcpyDeviceToHost));
    }

    // Free device memory for A, B, C
    CUDA_CHECK(cudaFree(dev_a)); 
    CUDA_CHECK(cudaFree(dev_b)); 
    CUDA_CHECK(cudaFree(dev_c));
}

void printArray(int **array,int WIDTH){
    int x,y;
    for(y=0;y<WIDTH;y++){
        for(x=0;x<WIDTH;x++)
            printf("%10d",array[y][x]);  
        printf("\n");
    }
}
