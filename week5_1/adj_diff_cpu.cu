#include <stdio.h>
#include <stdlib.h> // for rand(), malloc(), free()
#include <fcntl.h> // for open(), write()
#include <sys/stat.h>
#include "common.h"
#include <sys/time.h>
#define GRIDSIZE (64 * 1024)
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE * BLOCKSIZE) // 32M byte needed!

void genData(float* ptr, unsigned int size) {
    while (size--) {
        *ptr++ = (float)(rand() % 1000) / 1000.0F;
    }
}
// compute result[i] = input[i] – input[i-1]
void getDiff(float* dst, const float* src, unsigned int size) {
    for (int i = 1; i < size; ++i) {
        dst[i] = src[i] - src[i-1];
    }
}
int main(void) {
    float* pSource = NULL;
    float* pResult = NULL;
    int i;
    struct timeval start_time, end_time;
    // malloc memories on the host-side
    pSource = (float*)malloc(TOTALSIZE * sizeof(float));
    pResult = (float*)malloc(TOTALSIZE * sizeof(float));
    // generate source data
    genData(pSource, TOTALSIZE);
    // get current time
    gettimeofday(&start_time, NULL);
    getDiff(pResult, pSource, TOTALSIZE);
    // get end time
    gettimeofday(&end_time, NULL);
    double operating_time = (double)(end_time.tv_sec)+(double)(end_time.tv_usec)/1000000.0 -
    ((double)(start_time.tv_sec)+(double)(start_time.tv_usec)/1000000.0);
    printf("Elapsed: %f seconds\n", (double)operating_time);
    // print sample cases
    i = 1;
    printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
    i = TOTALSIZE - 1;
    printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
    i = TOTALSIZE / 2;
    printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
    // free the memory
    free(pSource);
    free(pResult);
}