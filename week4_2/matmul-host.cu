#include <cuda.h>
#include <iostream>

void matrixMul(int** A, int** B, int** C, int WIDTH); //loading, transfer, execution(host code)
void printArray(int **array, int WIDTH); // print Array elements
int main(int argc, char * argv[]){
    int WIDTH;
    int x,y;
    int **a,**b,**c;
    
    if(argc<2){
        printf("Not inserted block size, Try again\n");
        exit(-1);
    }
    //allocating program arguments
    WIDTH=atoi(argv[1]);

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
   matrixMul(a,b,c,WIDTH);
   printArray(c,WIDTH);
   printf("Matrix Multiplication execution time(Serial): %fms\n",(double)(clock()-start));
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


void matrixMul(int** A, int** B, int** C,int WIDTH)
{
    int x,y,k,sum;
    for(y=0;y<WIDTH;y++){
        for(x=0;x<WIDTH;x++){
            sum=0;
            for(k=0;k<WIDTH;k++){
                sum+=A[y][k]*B[k][x];
            }
            C[y][x]=sum;
        }
    }
}

void printArray(int **array, int WIDTH){
    int x,y;
    for(y=0;y<WIDTH;y++){
        for(x=0;x<WIDTH;x++)
            printf("%10d",array[y][x]);  
        printf("\n");
    }
}
