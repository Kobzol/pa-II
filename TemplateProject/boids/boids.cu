#include <vector>
#include <ctime>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>

#include "boids.h"
#include "../cudautil.cuh"
#include "../cudamem.h"
#include "../opengl/sceneManager.h"
#include "../opengl/demos/demo_boids.h"

#include <cuda_gl_interop.h>

#define BOID_COUNT (50)
#define TPB (BOID_COUNT)
#define VISUALIZE


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
struct Force
{
	float3 alignment;
	int alignmentCount;

	float3 cohesion;
	int cohesionCount;

	float3 separation;
	int separationCount;
};
struct FlockConfig
{
	float boidSeparationFactor = 1.0f;
	float boidCohesionFactor = 0.75f;
	float boidAlignmentFactor = 0.7f;
	float boidGoalFactor = 1 / 30.0f;

	float boidSeparateNearby = 1.0f;
	float boidCohesionNearby = 4.0f;
	float boidAlignmentNearby = 2.0f;

	float boidMaxVelocity = 0.01f;
};

static std::atomic<bool> run{ true };
static std::atomic<bool> configDirty{ true };
static std::mutex configMutex;
static FlockConfig flockConfig;

/// CUDA
__constant__ float3 cGoal;

static __device__ float vecLength(float3 vec)
{
	return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}
static __device__ float3 vecNormalize(float3 vec)
{
	float length = vecLength(vec);
	if (length == 0.0f) return vec;

	return vec / length;
}
static __device__ float3 vecClamp(float3 vec, float max)
{
	float length = vecLength(vec);
	if (length > max)
	{
		return vec * (max / length);
	}
	return vec;
}

static __device__ float3 updateSeparation(float3 position, float3 otherPosition, int& count, FlockConfig* config)
{
	float3 vec = position - otherPosition;
	float length = vecLength(vec);
	if (length == 0 || length >= config->boidSeparateNearby)
	{
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	count++;
	return vecNormalize(vec) / length;
}
static __device__ float3 updateCohesion(float3 position, float3 otherPosition, int& count, FlockConfig* config)
{
	float3 vec = position - otherPosition;
	float length = vecLength(vec);
	if (length == 0 || length >= config->boidCohesionNearby)
	{
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	count++;
	return otherPosition;
}
static __device__ float3 updateAlignment(float3 position, float3 otherPosition, float3 otherDirection, int& count, FlockConfig* config)
{
	float3 vec = position - otherPosition;
	float length = vecLength(vec);
	if (length == 0 || length >= config->boidAlignmentNearby)
	{
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	count++;
	return otherDirection;
}
/*
static __device__ void updateFlock(Force& force, Boid* sharedBoids, const float3& position, const int tileSize)
{
	for (int i = 0; i < tileSize; i++)
	{
		force.separation += updateSeparation(position, sharedBoids[i].position);
		force.cohesion += updateCohesion(sharedBoids[i].position, force.cohesionCount);
		force.alignment += updateAlignment(sharedBoids[i].direction, force.alignmentCount);
	}
}
*/
static __global__ void updateDirections(Boid* __restrict__ boids, float3* __restrict__ outDirections, const int size, FlockConfig* config)
{
#pragma region Init
	//__shared__ Boid sharedBoids[TPB];

	const int tileSize = blockDim.x;
	const int tileCount = gridDim.x;
	const int boidId = blockDim.x * blockIdx.x + threadIdx.x;

	if (boidId >= size) return;

	float3 position = boids[/*min(boidId, size - 1)*/boidId].position;
	Force force = { 0 };

	for (int i = 0; i < size; i++)
	{
		if (i != boidId)
		{
			force.separation += updateSeparation(position, boids[i].position, force.separationCount, config);
			force.cohesion += updateCohesion(position, boids[i].position, force.cohesionCount, config);
			force.alignment += updateAlignment(position, boids[i].position, boids[i].direction, force.alignmentCount, config);
		}
	}

	/*int boidsLeft = size;
#pragma endregion
	
#pragma region Traverse tiles
	for (int tile = 0; tile < tileCount - 1; tile++)
	{
		int tid = tile * tileSize + threadIdx.x;
		sharedBoids[threadIdx.x] = boids[tid];
		__syncthreads();

		updateFlock(force, sharedBoids, position, tileSize);
		boidsLeft -= tileSize;
		__syncthreads();
	}
#pragma endregion

#pragma region Traverse last tile
	int tid = (tileCount - 1) * tileSize + threadIdx.x;
	if (tid < size)
	{
		sharedBoids[threadIdx.x] = boids[tid];
	}
	__syncthreads();

	updateFlock(force, sharedBoids, position, boidsLeft);
	__syncthreads();
#pragma endregion*/

#pragma region  Create force vector
	if (force.cohesionCount > 0)
	{
		force.cohesion /= force.cohesionCount;	// center of mass
	}
	if (force.alignmentCount > 0)
	{
		force.alignment /= force.alignmentCount;
	}
	if (force.separationCount > 0)
	{
		force.separation /= force.separationCount;
	}

	float3 direction = make_float3(0.0f, 0.0f, 0.0f);
	direction += force.alignment * config->boidAlignmentFactor;
	direction += force.separation * config->boidSeparationFactor;
	direction += (force.cohesion - position) * config->boidCohesionFactor;
	direction += vecNormalize(cGoal - position) * config->boidGoalFactor;

	outDirections[boidId] = direction;
#pragma endregion
}
static __global__ void updatePositions(Boid* boids, float3* directions, size_t size, FlockConfig* config)
{
	const int boidId = blockDim.x * blockIdx.x + threadIdx.x;
	if (boidId >= size) return;

	boids[boidId].direction += directions[boidId];
	boids[boidId].direction = vecClamp(boids[boidId].direction, config->boidMaxVelocity);
	boids[boidId].position += boids[boidId].direction;
}

/// C++
static std::vector<Boid> init_boids(int count)
{
	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_real_distribution<float> posDist(0.0f, 1.0f);
	std::uniform_real_distribution<float> dirDist(0.01f, 0.01f);

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
static void copyTransformsToCuda(DemoBoids* demo, CudaMemory<Boid>& boids)
{
	std::vector<Boid> boidsCpu(BOID_COUNT);
	boids.load(*boidsCpu.data(), BOID_COUNT);

	for (int i = 0; i < boidsCpu.size(); i++)
	{
		glm::mat4 model;
		model = glm::translate(model, glm::vec3(boidsCpu[i].position.x, boidsCpu[i].position.y, boidsCpu[i].position.z));
		model = glm::scale(model, glm::vec3(0.1f, 0.1f, 0.1f));
		demo->models[i] = model;
	}
}

void boids_body(int argc, char** argv)
{
	srand((unsigned int)time(nullptr));

#ifdef VISUALIZE
	SceneManager* sceneManager = SceneManager::GetInstance();
	DemoBoids* demo = new DemoBoids(sceneManager->m_sceneData, BOID_COUNT);
	sceneManager->Init(argc, argv, demo);
	cudaGLSetGLDevice(0);
#endif

	float3 goal = make_float3(10.0f, 0.0f, 0.0f);
	CudaConstant<float3>::toDevice(cGoal, &goal);

	std::vector<Boid> boids = init_boids(BOID_COUNT);
	CudaMemory<Boid> cudaBoids(boids.size(), boids.data());
	CudaMemory<float3> outDirectionsCuda(BOID_COUNT);

	dim3 blockDim(TPB, 1);
	dim3 gridDim(getNumberOfParts(BOID_COUNT, TPB), 1);

	CudaMemory<FlockConfig> flockConfigCuda(1, &flockConfig);

	while (run)
	{
#ifdef VISUALIZE
		{
			if (configDirty)
			{
				std::lock_guard<decltype(configMutex)> lock(configMutex);
				flockConfigCuda.store(flockConfig);
				configDirty = false;
			}
		}
#endif

		CudaTimer timer;
		timer.start();
		updateDirections << <gridDim, blockDim >> > (cudaBoids.device(), outDirectionsCuda.device(), BOID_COUNT, flockConfigCuda.device());
		timer.stop_wait();
		//timer.print("Update directions: ");

		timer.start();
		updatePositions << <gridDim, blockDim >> > (cudaBoids.device(), outDirectionsCuda.device(), BOID_COUNT, flockConfigCuda.device());
		timer.stop_wait();
		//timer.print("Update positions: ");

		copyTransformsToCuda(demo, cudaBoids);

#ifdef VISUALIZE
		sceneManager->Refresh();
		Sleep(5);
#endif
	}
}
void boids(int argc, char** argv)
{
	std::thread runThread(boids_body, argc, argv);
	
	std::string line;
	while (std::getline(std::cin, line))
	{
		if (line[0] == 'q')
		{
			run = false;
			runThread.join();
			break;
		}
		else if (line[0] == 'a')
		{
			std::lock_guard<decltype(configMutex)> lock(configMutex);
			configDirty = true;
			flockConfig.boidCohesionFactor = -flockConfig.boidCohesionFactor;
		}
	}
}
