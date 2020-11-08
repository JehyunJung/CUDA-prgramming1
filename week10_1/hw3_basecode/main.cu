/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include "support.h"
#include "constants.cuh"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    Matrix M_h, N_h, P_h; // M: filter, N: input image, P: output image
    Matrix N_d, P_d;
    unsigned imageHeight, imageWidth;
    cudaError_t cuda_ret;
    dim3 dim_grid, dim_block;
    
    /* Read image dimensions */
    if (argc == 3) {
        imageHeight = atoi(argv[1]);
        imageWidth = atoi(argv[2]);
    } else {
        printf("\n    Invalid input parameters!"
	   "\n    Usage: ./convolution <m> <n>  # Image is m x n"
           "\n");
        exit(0);
    }

    /* Allocate host memory */
    M_h = allocateMatrix(FILTER_SIZE, FILTER_SIZE);
    N_h = allocateMatrix(imageHeight, imageWidth);
    P_h = allocateMatrix(imageHeight, imageWidth);

    /* Initialize filter and images */
    initMatrix(M_h);
    initMatrix(N_h);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Image: %u x %u\n", imageHeight, imageWidth);
    printf("    Mask: %u x %u\n", FILTER_SIZE, FILTER_SIZE);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    N_d = allocateDeviceMatrix(imageHeight, imageWidth);
    P_d = allocateDeviceMatrix(imageHeight, imageWidth);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    /* Copy image to device global memory */
    //INSERT CODE HERE
    cudaMemcpy(N_d.elements,N_h.elements,sizeof(float)*imageHeight*imageWidth,cudaMemcpyHostToDevice);
    cudaMemcpy(P_d.elements,P_h.elements,sizeof(float)*imageHeight*imageWidth,cudaMemcpyHostToDevice);

    /* Copy mask to device constant memory */
    //INSERT CODE HERE
    cudaMemcpyToSymbol(Mc,M_h.elements,sizeof(float)*FILTER_SIZE*FILTER_SIZE);
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    /*Launch kernel ---------------------------------------------------------*/
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    //INSERT CODE HERE
    dim_block=dim3(BLOCK_SIZE,BLOCK_SIZE);
    dim_grid=dim3(ceil((double)imageWidth/TILE_SIZE),ceil((double)imageHeight/TILE_SIZE),1);
    convolution<<<dim_grid,dim_block>>>(N_d,P_d);
    
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy data from device to host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    copyFromDeviceMatrix(P_h, P_d);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(M_h, N_h, P_h);

    // Free memory ------------------------------------------------------------

     freeMatrix(M_h);
     freeMatrix(N_h);
     freeMatrix(P_h);
     freeDeviceMatrix(N_d);
     freeDeviceMatrix(P_d);

     return 0;
}

