#include <vector>

#include "cudautil.cuh"
#include "cudamem.h"


#define TPB (128)

static __global__ void sumVector(float* __restrict__ a, float* __restrict__ b, size_t size, float* __restrict__ c)
{
	int tid = getGlobalIdx_1D_1D();

	const int threadCount = gridDim.x * blockDim.x;

	while (tid < size)
	{
		c[tid] = a[tid] + b[tid];
		tid += threadCount;
	}
}

static void vectors()
{
	size_t M = 1024 * 1024;
	std::vector<float> a, b;
	for (size_t i = 0; i < M; i++)
	{
		a.push_back(rand());
		b.push_back(rand());
	}

	CudaMemory<float> deviceA(a.size(), &a[0]);
	CudaMemory<float> deviceB(b.size(), &b[0]);
	CudaMemory<float> deviceC(a.size());

	dim3 dimBlock(TPB, 1, 1);
	dim3 dimGrid(getNumberOfParts(M, TPB), 1, 1);

	{
		CudaTimer timer(true);
		sumVector << <dimGrid, dimBlock >> > (deviceA.device(), deviceB.device(), a.size(), deviceC.device());
	}

	std::vector<float> c(a.size(), 0.0f);
	deviceC.load(c.at(0), c.size());

	for (size_t i = 0; i < c.size(); i++)
	{
		if (c[i] != a[i] + b[i])
		{
			throw "no match";
		}
	}
}
static void matrices()
{
	size_t N = 1000;
	size_t M = 1000;
	std::vector<float> a, b;
	for (size_t i = 0; i < N * M; i++)
	{
		a.push_back(rand());
		b.push_back(rand());
	}

	CudaMemory<float> deviceA(a.size(), &a[0]);
	CudaMemory<float> deviceB(b.size(), &b[0]);
	CudaMemory<float> deviceC(a.size());

	{
		CudaTimer timer(true);
		sumVector << <1, dim3(M, N, 1) >> > (deviceA.device(), deviceB.device(), a.size(), deviceC.device());
	}
	
	std::vector<float> c(a.size(), 0.0f);
	deviceC.load(c.at(0), c.size());
}

void cviko1()
{
	srand((unsigned int) NULL);

	vectors();
}
