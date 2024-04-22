/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Parallel reductio

    This sample shows how to perform a reduction operation on an array of values
    to produce a single value.

    Reductions are a very common computation in parallel algorithms.  Any time
    an array of values needs to be reduced to a single value using a binary
    associative operator, a reduction can be used.  Example applications include
    statistics computations such as mean and standard deviation, and image
    processing applications such as finding the total luminance of an
    image.

    This code performs sum reductions, but any associative operator such as
    min() or max() could also be used.

    It assumes the input size is a power of 2.

    COMMAND LINE ARGUMENTS
      -k [version]     0, 1, or 2

  */

// CUDA Runtime
#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <string.h>

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// includes, project
#include "reduction.h"

enum ReduceType { REDUCE_INT, REDUCE_FLOAT, REDUCE_DOUBLE };

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
template <class T>
bool runTest(int argc, char **argv, ReduceType datatype);

#define MAX_BLOCK_DIM_SIZE 65535

#ifdef WIN32
#define strcasecmp strcmpi
#endif

extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

const char *getReduceTypeString(const ReduceType type) {
  switch (type) {
    case REDUCE_INT:
      return "int";
    case REDUCE_FLOAT:
      return "float";
    case REDUCE_DOUBLE:
      return "double";
    default:
      return "unknown";
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("%s Starting...\n\n", argv[0]);


  ReduceType datatype = REDUCE_INT;


  cudaDeviceProp deviceProp;
  int dev = 0;

  //dev = findCudaDevice(argc, (const char **)argv);

  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

  printf("Using Device %d: %s\n\n", dev, deviceProp.name);
  checkCudaErrors(cudaSetDevice(dev));

  printf("Reducing array of type %s\n\n", getReduceTypeString(datatype));

  bool bResult = false;

  switch (datatype) {
    default:
    case REDUCE_INT:
      bResult = runTest<int>(argc, argv, datatype);
      break;

    case REDUCE_FLOAT:
      bResult = runTest<float>(argc, argv, datatype);
      break;

    case REDUCE_DOUBLE:
      bResult = runTest<double>(argc, argv, datatype);
      break;
  }

  printf(bResult ? "Test passed\n" : "Test failed!\n");
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template <class T>
T reduceCPU(T *data, int size) {
  T sum = data[0];
  T c = (T)0.0;

  for (int i = 1; i < size; i++) {
    T y = data[i] - c;
    T t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

#ifndef MIN
#define MIN(x, y) ((x < y) ? x : y)
#endif

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction
// kernel For the kernels >= 3, we set threads / block to the minimum of
// maxThreads and n/2. For kernels < 3, we set to the minimum of maxThreads and
// n.  For kernel 6, we observe the maximum specified number of blocks, because
// each thread in that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks,
                            int maxThreads, int &blocks, int &threads) {
  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));

  if (whichKernel < 3) {
    threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
    blocks = (n + threads - 1) / threads;
  } else {
    threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
  }

  if ((float)threads * blocks >
      (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
    printf("n is too large, please choose a smaller number!\n");
  }

  if (blocks > prop.maxGridSize[0]) {
    printf(
        "Grid size <%d> exceeds the device capability <%d>, set block size as "
        "%d (original %d)\n",
        blocks, prop.maxGridSize[0], threads * 2, threads);

    blocks /= 2;
    threads *= 2;
  }

  if (whichKernel >= 6) {
    blocks = MIN(maxBlocks, blocks);
  }
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
template <class T>
T benchmarkReduce(int n, int numThreads, int numBlocks, int maxThreads,
                  int maxBlocks, int whichKernel, int testIterations,
                  int cpuFinalThreshold,
                  double& time, T *h_odata, T *d_idata,
                  T *d_odata) {
  T gpu_result = 0;
  bool needReadBack = true;

  T *d_intermediateSums;
  checkCudaErrors(
      cudaMalloc((void **)&d_intermediateSums, sizeof(T) * numBlocks));

  for (int i = 0; i < testIterations; ++i) {
    gpu_result = 0;

    cudaDeviceSynchronize();
    double start = clock();
    //sdkStartTimer(&timer);

    // execute the kernel
    reduce<T>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);

    // check if kernel execution generated an error
    //getLastCudaError("Kernel execution failed");

    
    // sum partial block sums on GPU
    int s = numBlocks;
    int kernel = whichKernel;

    while (s > cpuFinalThreshold) {
      int threads = 0, blocks = 0;
      getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks,
                              threads);
      checkCudaErrors(cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(T),
                                  cudaMemcpyDeviceToDevice));
      reduce<T>(s, threads, blocks, kernel, d_intermediateSums, d_odata);

      if (kernel < 3) {
        s = (s + threads - 1) / threads;
      } else {
        s = (s + (threads * 2 - 1)) / (threads * 2);
      }
    }

    if (s > 1) {
      // copy result from device to host
      checkCudaErrors(cudaMemcpy(h_odata, d_odata, s * sizeof(T),
                                  cudaMemcpyDeviceToHost));

      for (int i = 0; i < s; i++) {
        gpu_result += h_odata[i];
      }

      needReadBack = false;
    }
    

    cudaDeviceSynchronize();
    time = clock() - start;
    //sdkStopTimer(&timer);
  }

  if (needReadBack) {
    // copy final sum from device to host
    checkCudaErrors(
        cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));
  }
  checkCudaErrors(cudaFree(d_intermediateSums));
  return gpu_result;
}



////////////////////////////////////////////////////////////////////////////////
// The main function which runs the reduction test.
////////////////////////////////////////////////////////////////////////////////
template <class T>
bool runTest(int argc, char **argv, ReduceType datatype) {
  int size = 1 << 24;    // number of elements to reduce
  int maxThreads = 256;  // number of threads per block
  int whichKernel = 0;
  int maxBlocks = 64;
  int cpuFinalThreshold = 1;

  int iarg = 1;
  while (iarg < argc) {
    if (strcmp(argv[iarg], "-k") == 0) {
      whichKernel = atoi(argv[iarg+1]);
      iarg += 2;
    } else {
      printf("invalid argument %s\n", argv[iarg]);
      iarg += 2;
    }

  }
  printf("%d elements\n", size);
  printf("%d threads (max)\n", maxThreads);
  printf("Kernel version %d\n", whichKernel);

  // create random input data on CPU
  unsigned int bytes = size * sizeof(T);

  T *h_idata = (T *)malloc(bytes);

  for (int i = 0; i < size; i++) {
    // Keep the numbers small so we don't get truncation error in the sum
    if (datatype == REDUCE_INT) {
      h_idata[i] = (T)(rand() & 0xFF);
    } else {
      h_idata[i] = (rand() & 0xFF) / (T)RAND_MAX;
    }
  }

  int numBlocks = 0;
  int numThreads = 0;
  getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks,
                          numThreads);

  if (numBlocks == 1) {
    cpuFinalThreshold = 1;
  }

  // allocate mem for the result on host side
  T *h_odata = (T *)malloc(numBlocks * sizeof(T));

  printf("%d blocks\n\n", numBlocks);

  // allocate device memory and data
  T *d_idata = NULL;
  T *d_odata = NULL;

  checkCudaErrors(cudaMalloc((void **)&d_idata, bytes));
  checkCudaErrors(cudaMalloc((void **)&d_odata, numBlocks * sizeof(T)));

  // copy data directly to device memory
  checkCudaErrors(
      cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks * sizeof(T),
                              cudaMemcpyHostToDevice));

  // warm-up
  reduce<T>(size, numThreads, numBlocks, whichKernel, d_idata, d_odata);

  int testIterations = 100;

  //StopWatchInterface *timer = 0;
  ///sdkCreateTimer(&timer);
  double elapsed_time;

  T gpu_result = 0;

  gpu_result =
      benchmarkReduce<T>(size, numThreads, numBlocks, maxThreads, maxBlocks,
                          whichKernel, testIterations,
                          cpuFinalThreshold, elapsed_time, h_odata, d_idata, d_odata);

  double reduceTime = elapsed_time / CLOCKS_PER_SEC; //sdkGetAverageTimerValue(&timer) * 1e-3;
  printf(
      "Reduction, Throughput = %.4f GB/s, Time = %.5f s, Size = %u Elements, "
      "NumDevsUsed = %d, Workgroup = %u\n",
      1.0e-9 * ((double)bytes) / reduceTime, reduceTime, size, 1, numThreads);

  // compute reference solution
  T cpu_result = reduceCPU<T>(h_idata, size);

  int precision = 0;
  double threshold = 0;
  double diff = 0;

  if (datatype == REDUCE_INT) {
    printf("\nGPU result = %d\n", (int)gpu_result);
    printf("CPU result = %d\n\n", (int)cpu_result);
  } else {
    if (datatype == REDUCE_FLOAT) {
      precision = 8;
      threshold = 1e-8 * size;
    } else {
      precision = 12;
      threshold = 1e-12 * size;
    }

    printf("\nGPU result = %.*f\n", precision, (double)gpu_result);
    printf("CPU result = %.*f\n\n", precision, (double)cpu_result);

    diff = fabs((double)gpu_result - (double)cpu_result);
    

    // cleanup
    //sdkDeleteTimer(&timer);
    free(h_idata);
    free(h_odata);

    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    if (datatype == REDUCE_INT) {
      return (gpu_result == cpu_result);
    } else {
      return (diff < threshold);
    }
  }

  return true;
}
