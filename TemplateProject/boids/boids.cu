#include <vector>
#include <ctime>
#include <random>

#include "boids.h"
#include "../cudautil.cuh"
#include "../cudamem.h"
#include "../opengl/sceneManager.h"
#include "../opengl/demos/demo_boids.h"


/// Structures
struct Boid
{
	Boid()
	{

	}
	Boid(float3 position, float3 direction) : position(position), direction(direction)
	{

	}

	float3 position;
	float3 direction;
};

/// CUDA
__constant__ float3 cGoal;

static __global__ void updateDirections(Boid* boids, size_t size)
{
	__shared__ Boid sharedBoids[TPB];

	const int tileSize = blockDim.x;
	const int tileCount = gridDim.x;
	const int boidId = blockDim.x * blockIdx.x + threadIdx.x;

	if (boidId >= size) return;

	float3 direction = boids[boidId].direction;

	// traverse all tiles
	for (int tile = 0; tile < tileCount; tile++)
	{
		// copy boids to shared memory
		int tid = tile * tileSize + threadIdx.x;
		sharedBoids[threadIdx.x] = boids[tid];
		__syncthreads();

		// update boid velocity
		for (int i = 0; i < tileSize; i++)
		{
			direction += sharedBoids[i].direction;
		}

		__syncthreads();
	}

	boids[boidId].direction = direction + cGoal;
}
static __global__ void updatePositions(Boid* boids, size_t size)
{
	const int boidId = blockDim.x * blockIdx.x + threadIdx.x;
	if (boidId >= size) return;

	boids[boidId].position += boids[boidId].direction;
}

/// C
static std::vector<Boid> init_boids(int count)
{
	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_real_distribution<float> posDist(0.0f, 1.0f);
	std::uniform_real_distribution<float> dirDist(-2.0f, 2.0f);

	std::vector<Boid> boids;
	for (int i = 0; i < count; i++)
	{
		boids.emplace_back(
			make_float3(posDist(engine), posDist(engine), posDist(engine)),
			make_float3(dirDist(engine), dirDist(engine), dirDist(engine))
		);
	}

	return boids;
}

void boids(int argc, char** argv)
{
	srand((unsigned int) time(nullptr));

	float3 goal = make_float3(10.0f);
	CudaConstant<float3>::toDevice(cGoal, &goal);

	std::vector<Boid> boids = init_boids(BOID_COUNT);
	CudaMemory<Boid> cudaBoids(boids.size(), boids.data());

	dim3 blockDim(TPB, 1);
	dim3 gridDim(getNumberOfParts(BOID_COUNT, TPB), 1);

#ifdef VISUALIZE
	SceneManager* sceneManager = SceneManager::GetInstance();
	DemoBoids* demo = new DemoBoids(sceneManager->m_sceneData, BOID_COUNT);
	sceneManager->Init(argc, argv, demo);
#endif

	while (true)
	{
		CudaTimer timer;
		timer.start();
		updateDirections << <gridDim, blockDim >> > (cudaBoids.device(), BOID_COUNT);
		timer.stop_wait();
		timer.print("Update directions: ");

		timer.start();
		updatePositions << <gridDim, blockDim >> > (cudaBoids.device(), BOID_COUNT);
		timer.stop_wait();
		timer.print("Update positions: ");

#ifdef VISUALIZE
		sceneManager->Refresh();
#endif
	}
}
