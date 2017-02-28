
export LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

nvcc -arch=sm_35 -I/usr/local/cuda/include -I/usr/local/NVIDIA_CUDA-7.5_Samples/common/inc -L./  -L /usr/local/cuda/lib64/ -lcudart -L/usr/lib64  deviceQuery.cpp
