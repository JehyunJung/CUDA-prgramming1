#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>
#include "common.h"
#define CHANNELS 3 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
__global__ void colorConvert(unsigned char * grayImage,  unsigned char * rgbImage,int width, int height) {
 int x = threadIdx.x + blockIdx.x * blockDim.x;
 int y = threadIdx.y + blockIdx.y * blockDim.y;

 //check whether the threads with both Row and Col are within range 
 if (x < width && y < height) {
    // get 1D coordinate for the grayscale image
    int grayOffset = y*width + x;
    
    // one can think of the RGB image having
    // CHANNEL times columns than the gray scale image
    int rgbOffset = grayOffset*CHANNELS;
    unsigned char r =  rgbImage[rgbOffset      ]; // red value for pixel
    unsigned char g = rgbImage[rgbOffset + 1]; // green value for pixel
    unsigned char b = rgbImage[rgbOffset + 2]; // blue value for pixel
    
    // perform the rescaling and store it
    // We multiply by floating point constants
    grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
 }
}
int main(int argc, char **argv)
{
	std::string rgb_filename;
	std::string grey_filename;
	unsigned int pixels=0;
	//Check for the input file and output file name
	if(argc==3){
		rgb_filename = std::string(argv[1]);
		grey_filename= std::string(argv[2]);
	}else{
		std::cerr << "Usage: <executable> rgb_filename grey_filename";
		exit(1);
	}
	// host and device’s image pointers
	unsigned char *rgb_d, *rgb_h; 
	unsigned char *grey_d, *grey_h; 

	int rows; //number of rows of pixels
	int cols; //number of columns of pixels

	//load image into an array and retrieve number of pixels
	cv::Mat image;
	image = cv::imread(rgb_filename.c_str(), CV_LOAD_IMAGE_COLOR);
	

	if(image.empty())
	{
		std::cerr <<"fail to open image file"<<std::endl;
		exit(1);
	}
	rows=image.rows;
	cols=image.cols;

	rgb_h = (unsigned char*) malloc(rows*cols*sizeof(unsigned char)*3);
	unsigned char* rgb_data=(unsigned char*)image.data;
	pixels = rows*cols;
	for (int i=0;i<pixels*CHANNELS;i++)
	{	
   		rgb_h[i]=rgb_data[i];
	}


	//allocate host memory for grey image
	grey_h= (unsigned char *)malloc(sizeof(unsigned char)*pixels);

	//allocate and initialize memory on device
	CUDA_CHECK(cudaMalloc(&rgb_d, sizeof(unsigned char) * pixels * CHANNELS));
	CUDA_CHECK(cudaMalloc(&grey_d, sizeof(unsigned char) * pixels));
	CUDA_CHECK(cudaMemset(grey_d, 0, sizeof(unsigned char) * pixels));

	//copy rgb image from host to device
	CUDA_CHECK(cudaMemcpy(rgb_d, rgb_h, sizeof(unsigned char)*pixels*CHANNELS, cudaMemcpyHostToDevice));

	//define block and grid dimensions
	const dim3 dimGrid(ceil(cols/16), ceil(rows/16),1);
	const dim3 dimBlock(16,16);

	//execute cuda kernel
	colorConvert<<<dimGrid, dimBlock>>>(grey_d, rgb_d, cols, rows);
	CUDA_CHECK( cudaPeekAtLastError() );

	//copy computed gray image from device to host
	CUDA_CHECK(cudaMemcpy(grey_h, grey_d, sizeof(unsigned char) * pixels, cudaMemcpyDeviceToHost));

	//store the grey image
        cv::Mat greyData(rows, cols, CV_8UC1, (void *) grey_h);
	//write mAT OBJECT TO FILE
	cv::imwrite(grey_filename.c_str(), greyData);


	//free memory
	cudaFree(rgb_d);
	cudaFree(rgb_h);
 }



