/**
 * Vector scale: x <= x * k
 *
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
//#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Multiplies all the elements of vector x by a scalar value
 */
__global__ void scale(float *x, const float k, const int N) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < N) {
    x[tid] = x[tid] * k;
  }
}

/**
 * Host main routine
 */
int main(void) {

  float k = 10.0;

  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("Vector scaling of %d elements: k = %f\n", numElements, k);

  // Allocate the host input vector x
  float *h_x = (float *)malloc(size);

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_x[i] = i;
  }

  // Allocate the device input vector x
  float *d_x = NULL;
  cudaMalloc((void **)&d_x, size);

  // Copy the host input vector x in host memory to the device input
  // vectors in device memory
  
  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  scale<<<blocksPerGrid, threadsPerBlock>>>(d_x, k, numElements);

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_x[i]  - k * i) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  cudaFree(d_x);

  // Free host memory
  free(h_x);

  printf("Done\n");
  return 0;
}
