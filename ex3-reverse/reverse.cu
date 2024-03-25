//#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void reverse_kernel(int *d, int n)
{
  extern __shared__ int s[];

  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

int main(int argc, char** argv)
{
  // parse command line arguments
  if (argc != 2) {
    printf("Usage: ./a.out [Number of elements] \n");
    return 0;
  }

  const int n = atoi(argv[1]);
  int a[n], r[n], d[n];

  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1; // correct values for testing
    d[i] = 0;
  }

  printf("In-place reverse an array with %d elements..\n",  n);

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int));

  int passed = 1;

  // copy from a from HOST to DEVICE
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);

  // launch the kernel
  reverse_kernel<<<1, n, n*sizeof(int)>>>(d_d, n);

  // copy from d from DEVICE to HOST
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);

  // check the result
  passed = 1;
  for (int i = 0; i < n; i++) {
    if (d[i] != r[i]) {
       printf("Error: d[%d] != r[%d] (%d, %d)\n", i, i, d[i], r[i]);
       passed = 0;
       break;
    }
  }

  cudaFree(d_d);

  if (passed == 1) printf("Test PASSED\n");
  else printf("Test FAILED\n");

  return 0;
}