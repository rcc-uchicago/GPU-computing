# GPU-computing

This repo contains source code for the GPU computing workshop.

The example source files are based on the CUDA SDK Samples: 

https://github.com/NVIDIA/cuda-samples

To build the source codes (the `.cu` files in each folder) on Midway3, do

```
module load cuda/11.5 openmpi/4.1.2+gcc-10.2.0
```

and then go to each folder and do

```
cd ex1-scale
nvcc -O2 -arch=sm_70 scale.cu -o vec_scale
```
where `sm_70` means that you want the executable `vec_scale` to run on NVIDIA V100 GPUs.

To run the executables, you need a GPU node with at least a GPU:

```
sinteractive -A [your-allocation] --reservation=[reservation-name-if-needed] -p gpu --gres=gpu:1
```

Once on the compute node, load the modules again (if they are not loaded) before running the executables

```
module load cuda/11.5 openmpi/4.1.2+gcc-10.2.0
cd ex1-scale
./vec_scale
```


