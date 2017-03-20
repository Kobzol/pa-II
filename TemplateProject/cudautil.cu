#include "cudautil.cuh"

#include <cassert>

void checkCudaCall(cudaError_t error, const char* expression, const char* file, int line)
{
	if (error)
	{
		std::cout << "CUDA error at " << file << ":" << line << " - " << expression << std::endl;
		std::cout << cudaGetErrorName(error) << " :: " << cudaGetErrorString(error) << std::endl;
		assert(0);
	}
}

__device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}
__device__ int getGlobalIdx_1D_2D()
{
	return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}
__device__ int getGlobalIdx_2D_1D()
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}
__device__ int getGlobalIdx_2D_2D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}
