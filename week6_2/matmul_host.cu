#include <cuda.h>
#include <iostream>

void genData(float * ptr, unsigned int size);//random data generation
void matrixMul(float * A, float * B, float * C, int HEIGHT,int WIDTH, int COMMON); //loading, transfer, execution(host code)
void printMatrix(float * matrix, int HEIGHT, int WIDTH);//print matrix

int main(int argc, char * argv[]){
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
    genData(pA, HEIGHT * COMMON);
    genData(pB, COMMON * WIDTH);
    clock_t start=clock();
    
    matrixMul(pA,pB,pC,HEIGHT,WIDTH,COMMON);
    printMatrix(pC,HEIGHT,WIDTH);
    printf("Matrix Multiplication execution time(Serial): %fms\n",(double)(clock()-start));
  
    //free 2D array
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

void matrixMul(float* A, float* B, float* C,int HEIGHT,int WIDTH,int COMMON)
{
    int x,y,k;
    float sum=0.0F;
    for(y=0;y<HEIGHT;y++){
        for(x=0;x<WIDTH;x++){
            sum=0;
            for(k=0;k<COMMON;k++){
                sum+=A[y*COMMON+k]*B[k*WIDTH+x];
            }
            C[y*WIDTH+x]=sum;
        }
    }
}

void printMatrix(float * matrix, int HEIGHT, int WIDTH){
    int i,j;
    FILE *fp=fopen("output(cpu).txt","wt");
    for(i=0;i<HEIGHT;i++)
        for(j=0;j<WIDTH;j++)
            fprintf(fp,"c[%4d][%4d] = %f\n", i, j, matrix[i * WIDTH + j]);
}
