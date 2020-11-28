 #ifdef DEBUG // debug mode
 #define CUDA_CHECK(x)   do {\
     (x); \
     cudaError_t e = cudaGetLastError(); \
     if (cudaSuccess != e) { \
         printf("cuda failure %s at %s:%d\n", \
              cudaGetErrorString(e), \
              __FILE__, __LINE__); \
         exit(1); \
     } \
 } while (0)
 #else 
 #define CUDA_CHECK(x)   (x) // release mode
 #endif
 

