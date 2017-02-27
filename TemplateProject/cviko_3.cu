#include <vector>
#include <memory>
#include <ctime>

#include "cudautil.cuh"

#define TPB_REDUCE 128
#define TPB_APPLY 256
#define NO_FORCES 256
#define NO_RAIN_DROPS 1048576

#define MEM_BLOCKS_PER_THREAD_BLOCK	8

#define REDUCE(next, tid, mem) if (tid >= next) return; \
mem[tid] += mem[tid + next];

static __global__ void reduceWindmills(const float3* __restrict__ forces, const unsigned int noForces, float3* __restrict__ finalForce)
{
	__shared__ float3 sForces[TPB_REDUCE];
	unsigned int tid = threadIdx.x;
	unsigned int next = TPB_REDUCE;

	sForces[tid] = forces[tid] + forces[tid + next];	// load first level from global to shared
	next /= 2;

	__syncthreads();

	while (next > 32 && tid < next)	// load from previous level of shared memory
	{
		sForces[tid] += sForces[tid + next];
		next /= 2;
		__syncthreads();
	}

	// we are running in a warp, no need to synchronize
	REDUCE(32, tid, sForces);
	REDUCE(16, tid, sForces);
	REDUCE(8, tid, sForces);
	REDUCE(4, tid, sForces);
	REDUCE(2, tid, sForces);
	REDUCE(1, tid, sForces);

	// store the result
	if (tid == 0)
	{
		*finalForce = sForces[0];
	}
}
static __global__ void applyForces(float3* __restrict__ drops, const float3* __restrict__ windmill, const unsigned int dropsCount, const unsigned int memBlocks)
{
	unsigned int start = blockIdx.x * (memBlocks * blockDim.x);
	for (int i = 0; i < memBlocks; i++)
	{
		const int index = start + threadIdx.x + i * blockDim.x;
		drops[index] += *windmill;
	}
}

static void generateVectors(float3* data, size_t count)
{
	for (size_t i = 0; i < count; i++)
	{
		data[i] = make_float3(1.0f, 1.0f, 1.0f);
		/*data[i].x = rand() / 10.0f;
		data[i].y = rand() / 10.0f;
		data[i].z = rand() / 10.0f;*/
	}
}
static float3 reduceCpu(float3* data, size_t count)
{
	float3 result = { 0, 0, 0 };

	for (size_t i = 0; i < count; i++)
	{
		result.x += data[i].x;
		result.y += data[i].y;
		result.z += data[i].z;
	}

	return result;
}
static void applyCpu(float3* data, float3 vector, size_t count)
{
	for (size_t i = 0; i < count; i++)
	{
		data[i] += vector;
	}
}
static void compareVectors(float3* data, CudaMemory<float3>& cudaData, size_t count)
{
	std::unique_ptr<float3[]> localData = std::make_unique<float3[]>(count);
	cudaData.load(*localData.get(), count);

	for (int i = 0; i < count; i++)
	{
		if (data[i].x != localData[i].x)
		{
			throw "error";
		}
		if (data[i].y != localData[i].y)
		{
			throw "error";
		}
		if (data[i].z != localData[i].z)
		{
			throw "error";
		}
	}
}

static std::ostream& operator<<(std::ostream& os, float3& vec)
{
	os << "[" << vec.x << ", " << vec.y << ", " << vec.z << "]";
	return os;
}

static void raindrops()
{
	std::unique_ptr<float3[]> windmills = std::make_unique<float3[]>(NO_FORCES);
	generateVectors(windmills.get(), NO_FORCES);
	CudaMemory<float3> windmillsCuda(NO_FORCES, windmills.get());
	CudaMemory<float3> resultForceCuda;

	std::unique_ptr<float3[]> drops = std::make_unique<float3[]>(NO_RAIN_DROPS);
	generateVectors(drops.get(), NO_RAIN_DROPS);
	CudaMemory<float3> dropsCuda(NO_RAIN_DROPS, drops.get());

	dim3 applyBlockSize(TPB_APPLY, 1);
	dim3 applyGridSize(getNumberOfParts(NO_RAIN_DROPS, TPB_APPLY * MEM_BLOCKS_PER_THREAD_BLOCK), 1);

	float reduceTime = 0.0f, applyTime;
	const int iterations = 1000;

	for (int i = 0; i < iterations; i++)
	{
		CudaTimer timer;
		timer.start();
		reduceWindmills << <dim3(1, 1), dim3(TPB_REDUCE, 1) >> > (windmillsCuda.device(), NO_FORCES, resultForceCuda.device());
		timer.stop_wait();
		reduceTime += timer.get_time();

		float3 resultForce;
		resultForceCuda.load(resultForce);

		timer.start();
		applyForces << <applyGridSize, applyBlockSize>> > (dropsCuda.device(), resultForceCuda.device(), NO_RAIN_DROPS, MEM_BLOCKS_PER_THREAD_BLOCK);
		timer.stop_wait();
		applyTime += timer.get_time();

		applyCpu(drops.get(), resultForce, NO_RAIN_DROPS);
		compareVectors(drops.get(), dropsCuda, NO_RAIN_DROPS);
	}

	std::cerr << "Reduce: " << reduceTime << std::endl;
	std::cerr << "Apply: " << applyTime << std::endl;
}

void cviko3()
{
	srand((unsigned int) time(NULL));

	raindrops();
}
