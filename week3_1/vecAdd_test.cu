#include <cuda.h>
#include <iostream>
#include <time.h>
__global__
void addKernel(int* A_d, int* B_d, int*C_d); //vector addition(device code)
void arrayAdd(int* A, int* B, int* C, int n); //vector addition(serial code)
void vecAdd(int* A, int* B, int* C, int n); //loading, transfer, execution(host code)
void printArray(int * array,int n); // print Array elements
int main(void){
    // const int SIZE=5;
    // int a[SIZE]={1,2,3,4,5};
    // int b[SIZE]={10,20,30,40,50};
    // int c[SIZE]={0};
    int SIZE;
    int i;
    int *a,*b,*c;
    
    printf("Insert size: ");
    scanf("%d",&SIZE);

    a=(int*)malloc(sizeof(int)*SIZE);
    b=(int*)malloc(sizeof(int)*SIZE);
    c=(int*)malloc(sizeof(int)*SIZE);

    for(i=0;i<SIZE;i++){
        a[i]=i;
        b[i]=10*i;
    }

    clock_t start;
    
    start=clock();
    arrayAdd(a,b,c,SIZE);
    printf("Vector addition execution time(Serial): %fms\n",(double)(clock()-start));

    start=clock();
    vecAdd(a,b,c,SIZE);
    printf("Vector addition execution time(Parallel): %fms\n",(double)(clock()-start));
    printArray(c,SIZE);

    free(a);
    free(b);
    free(c);
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

void arrayAdd(int* A, int* B, int* C, int n){
    int i;
    for(i=0;i<n;i++)
        C[i]=A[i]+B[i];
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

void printArray(int * array,int n){
    int i,sum=0;
    printf("Elements: ");
    for(i=0;i<n;i++){
        printf("%d ",array[i]);
        sum+=array[i];
    }
    printf("\n");
    printf("Sum: %d\n",sum);
}