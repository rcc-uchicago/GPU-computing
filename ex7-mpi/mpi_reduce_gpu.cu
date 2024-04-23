/*
  module load openmpi cuda
  export CUDA_PATH=$CUDA_HOME
  export MPI_HOME=$OPENMPI_HOME
  make
*/

#include <stdio.h>
#include <mpi.h>
__global__ void square(int *d)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    d[i] = d[i] * d[i];
}
int main(int argc, char *argv[])
{
    // Initialize MPI.
    MPI_Init(&argc, &argv);
    // Get the node count and node rank.
    int commSize, commRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    // Get the number of CUDA devices.
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    // Initialize constants.
    int numThreadsPerBlock = 256;
    int numBlocksPerGrid = 10000;
    int dataSizePerNode = numThreadsPerBlock * numBlocksPerGrid;
    int dataSize = dataSizePerNode * commSize;
    // Generate some random numbers on the root node.
    int *data;
    if (commRank == 0)
    {
        data = new int[dataSize];
        for (int i = 0; i < dataSize; ++i)
        {
            data[i] = rand() % 10;
        }
    }
    // Allocate a buffer on the current node.
    int *dataPerNode = new int[dataSizePerNode];
    // Dispatch a portion of the input data to each node.
    MPI_Scatter(data, dataSizePerNode, MPI_INT, dataPerNode, dataSizePerNode, MPI_INT, 0, MPI_COMM_WORLD);
    // Compute the square of each element on device.
    int *d;
    cudaSetDevice(commRank % numDevices);
    cudaMalloc((void **)&d, sizeof(int) * dataSizePerNode);
    cudaMemcpy(d, dataPerNode, sizeof(int) * dataSizePerNode, cudaMemcpyHostToDevice);
    square<<<numBlocksPerGrid, numThreadsPerBlock>>>(d);
    cudaMemcpy(dataPerNode, d, sizeof(int) * dataSizePerNode, cudaMemcpyDeviceToHost);
    cudaFree(d);
    // Compute the sum of the current node.
    int sum = 0;
    for (int i = 0; i < dataSizePerNode; ++i)
    {
        sum += dataPerNode[i];
    }
    // Compute the sum of all nodes.
    int actual;
    MPI_Reduce(&sum, &actual, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // Validate the result.
    if (commRank == 0)
    {
        int expected = 0;
        for (int i = 0; i < dataSize; ++i)
        {
            expected += data[i] * data[i];
        }
        if (actual != expected)
        {
            printf("actual = %d, expected = %d\n", actual, expected);
        }
        delete[] data;
    }
    // Cleanup.
    delete[] dataPerNode;
    cudaDeviceReset();
    MPI_Finalize();
}
