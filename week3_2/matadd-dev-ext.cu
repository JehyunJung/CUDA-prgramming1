#include <cuda.h>
#include <iostream>
#include "common.h"

__global__ void addKernel(const int* dev_a, const int* dev_b, int* dev_c, int WIDTH,int SIZE); //matrix addition(device code)
void matrixAdd(int** A, int** B, int** C, int HEIGHT, int WIDTH); //loading, transfer, execution(host code)
void printArray(int **array,int HEIGHT, int WIDTH); // print Array elements
int main(int argc, char * argv[]){
    int HEIGHT, WIDTH;
    int x,y;
    int **a,**b,**c;
    
    if(argc<3){
        printf("Not inserted block size, Try again\n");
        exit(-1);
    }
    //allocating program arguments
    HEIGHT=atoi(argv[1]);
    WIDTH=atoi(argv[2]);

    //2D array dynamic allocation 
    a=(int**)malloc(sizeof(int*)*HEIGHT);
    b=(int**)malloc(sizeof(int*)*HEIGHT);
    c=(int**)malloc(sizeof(int*)*HEIGHT);

    for(y=0;y<HEIGHT;y++){
        a[y]=(int*)malloc(sizeof(int)*WIDTH);
        b[y]=(int*)malloc(sizeof(int)*WIDTH);
        c[y]=(int*)malloc(sizeof(int)*WIDTH);
    }
    
    //initializing 2D array
   for(y=0;y<HEIGHT;y++){
       for(x=0;x<WIDTH;x++){
           a[y][x]=y*10+x;
           b[y][x]=(y*10+x)*100;
        }
   }

   matrixAdd(a,b,c,HEIGHT,WIDTH);
   printArray(c,HEIGHT,WIDTH);

   //free 2D array
    for(y=0;y<HEIGHT;y++){
        free(a[y]);
        free(b[y]);
        free(c[y]);
    }

    free(a);
    free(b);
    free(c);
    return 0;
}

// Compute vector sum C = A+B
// Each thread performs one pairwise addition
__global__ void addKernel(const int* dev_a, const int* dev_b, int*dev_c, int WIDTH,int SIZE)
{
    int x = (blockIdx.x*blockDim.x)+threadIdx.x; //global index of x
    int y = (blockIdx.y*blockDim.y)+threadIdx.y; //global index of y
    int i = y * WIDTH + x;   //actual index of i(1D array)
    //error handling
    if(i<SIZE)
        dev_c[i] = dev_a[i] + dev_b[i];
}

void matrixAdd(int** A, int** B, int** C, int HEIGHT,int WIDTH)
{
    int* dev_a=0;
    int* dev_b=0;
    int* dev_c=0;
    int y=0;
    int SIZE=HEIGHT*WIDTH;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&dev_a, SIZE*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_b, SIZE*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&dev_c, SIZE*sizeof(int)));

    // Transfer A and B to device memory (need flattening)
    for(y=0;y<HEIGHT;y++){
        CUDA_CHECK(cudaMemcpy(&dev_a[y*WIDTH], A[y], WIDTH*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&dev_b[y*WIDTH], B[y], WIDTH*sizeof(int), cudaMemcpyHostToDevice));
    }
    
    //configure grid ==> <<<number of thread blocks within grid, number of threads in each thread block>>> 
    addKernel<<<ceil(WIDTH*HEIGHT/256.0), 256>>>(dev_a, dev_b, dev_c, WIDTH,SIZE);

    // Transfer C from device to host
    for(y=0;y<HEIGHT;y++){
        CUDA_CHECK(cudaMemcpy(C[y], &dev_c[y*WIDTH], WIDTH*sizeof(int), cudaMemcpyDeviceToHost));
    }

    // Free device memory for A, B, C
    CUDA_CHECK(cudaFree(dev_a)); 
    CUDA_CHECK(cudaFree(dev_b)); 
    CUDA_CHECK(cudaFree(dev_c));
}

void printArray(int **array,int HEIGHT, int WIDTH){
    int x,y;
    for(y=0;y<HEIGHT;y++){
        for(x=0;x<WIDTH;x++)
            printf("%5d",array[y][x]);  
        printf("\n");
    }
}
