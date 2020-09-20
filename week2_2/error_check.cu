#include <iostream>
#ifdef DEBUG
#define CUDA_CHECK(x) do {\
    (x); \
    cudaError_t e = cudaGetLastError(); \
    if (cudaSuccess != e) { \
            printf("cuda failure \"%s\" at %s:%d\n", \
            cudaGetErrorString(e), \
            __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
#else
#define CUDA_CHECK(x) (x)
#endif

// main program for the CPU
int main(void) {
    // host-side data
    const int SIZE = 5;
    const int a[SIZE] = { 1, 2, 3, 4, 5 };
    int b[SIZE] = { 0, 0, 0, 0, 0 };
    // print source
    printf("a = {%d,%d,%d,%d,%d}\n", a[0], a[1], a[2], a[3], a[4]);
    // device-side data
    int *dev_a = 0;
    int *dev_b = 0;
    // allocate device memory
    CUDA_CHECK( cudaMalloc((void**)&dev_a, SIZE * sizeof(int)) );
    CUDA_CHECK( cudaMalloc((void**)&dev_b, SIZE * sizeof(int)) );
    // copy from host to device
    CUDA_CHECK( cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyDeviceToDevice) ); // BOMB here !
    // copy from device to device
    CUDA_CHECK( cudaMemcpy(dev_b, dev_a, SIZE * sizeof(int), cudaMemcpyDeviceToDevice) );
    // copy from device to host
    CUDA_CHECK( cudaMemcpy(b, dev_b, SIZE * sizeof(int), cudaMemcpyDeviceToHost) );
    // free device memory
    CUDA_CHECK( cudaFree(dev_a) );
    CUDA_CHECK( cudaFree(dev_b) );
    // print the result
    printf("b = {%d,%d,%d,%d,%d}\n", b[0], b[1], b[2], b[3], b[4]);
    // done
    return 0;
}


