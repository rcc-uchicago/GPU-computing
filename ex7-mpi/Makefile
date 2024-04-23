# Mixing MPI and CUDA C
#
# NOTE: I need NVCC to compile CUDA code _but_ since all is mixed I also
# to link MPI properly because I am not using mpicc wrapper
# need CUDA_HOME defined (module show cuda)
# need MPI_HOME defined (module show openmpi or intelmpi or mpich)

NVCC=nvcc
NVCCFLAGS=-O2 -arch=sm_80 --ptxas-options=-v -I${CUDA_HOME}/include  -L${CUDA_HOME}/lib64 -lcudart

MPIHOME=${MPI_HOME}
MPIFLAGS=-I$(MPIHOME)/include -L$(MPIHOME)/lib -lmpi

default: mpi_reduce_gpu.x

clean:
	rm -f *.o *x out* *.o*

mpi_reduce_gpu.x: mpi_reduce_gpu.cu
	$(NVCC) -o $@ $(FLAGS) $(MPIFLAGS) $(NVCCFLAGS) $^
