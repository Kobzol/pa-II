#include <vector>
#include <ctime>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>

#include "../cudautil.cuh"
#include "../cudamem.h"

#include "boids.h"
#include "../opengl/sceneManager.h"
#include "../opengl/demos/demo_boids.h"
#include "../opengl/CoreHeaders/sceneGUI.h"


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <cuda_gl_interop.h>

#define BOID_COUNT (1)
#define THREADS_PER_BLOCK (256)

#define USE_SHARED_MEM
#define CHECK_VIEW_RANGE

#define VISUALIZE
//#define SIMULATE

double boidsSeparationFactor = 1.0;
double boidsCohesionFactor = 0.7;
double boidsAlignmentFactor = 1.0;
double boidsGoalFactor = 0.2;
glm::vec3 boidGoal{ 10.0f, 0.0f, 0.0f };

double boidsSeparationNeighbourhood = 1.0f;
double boidsCohesionNeighbourhood = 4.0f;
double boidsAlignmentNeighbourhood = 1.0f;

double boidsMaxVelocity = 0.01f;
double boidsViewAngle = 135.0f;

static glm::vec3 flockCenter{ 0.0f, 0.0f, 0.0f };

/// Test
double boidTestDir[3] = { 0.0, 0.0, 1.0 };

/// CUDA
static __device__ bool operator==(const float3& vec1, const float3& vec2)
{
	return vec1.x == vec2.x && vec1.y == vec2.y && vec1.z == vec2.z;
}

static __device__ float3 vecClamp(const float3& vec, float max)
{
	float len = length(vec);
	if (len != 0.0f && len > max)
	{
		return vec * (max / len);
	}
	return vec;
}
static __device__ float3 vecNormalize(const float3& vec)
{
	float len = length(vec);
	if (len == 0.0f) return vec;

	return vec / len;
}

static __device__ float3 updateSeparation(const float3& position, const float3& otherPosition, int& count, FlockConfig* config)
{
	float3 vec = position - otherPosition;
	float len = length(vec);
	if (len == 0 || len >= config->separationNeighbourhood)
	{
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	count++;
	return vecNormalize(vec) / len;
}
static __device__ float3 updateCohesion(const float3& position, const float3& otherPosition, int& count, FlockConfig* config)
{
	float3 vec = position - otherPosition;
	float len = length(vec);
	if (len == 0 || len >= config->cohesionNeighbourhood)
	{
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	count++;
	return otherPosition;
}
static __device__ float3 updateAlignment(const float3& position, const float3& otherPosition, const float3& otherDirection, int& count, FlockConfig* config)
{
	float3 vec = position - otherPosition;
	float len = length(vec);
	if (len == 0 || len >= config->alignmentNeighbourhood)
	{
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	count++;
	return otherDirection;
}

static __device__ void updateFlock(Force& force, const float3& position, const float3& otherPosition, const float3& otherDirection, FlockConfig* config)
{
	force.separation += updateSeparation(position, otherPosition, force.separationCount, config);
	force.cohesion += updateCohesion(position, otherPosition, force.cohesionCount, config);
	force.alignment += updateAlignment(position, otherPosition, otherDirection, force.alignmentCount, config);
}
static __device__ bool isInViewRange(const float3& position, const float3& direction, const float3& otherPosition, float viewAngle)
{
#ifdef CHECK_VIEW_RANGE
	if (position == otherPosition) return false;

	float3 toTarget = vecNormalize(otherPosition - position);

	float angle = atan2(length(cross(toTarget, direction)), dot(toTarget, direction));
	return angle < viewAngle;
#else
	return true;
#endif
}

static __global__ void calculateAccelerations(Boid* __restrict__ boids, Acceleration* __restrict__ outAccelerations, const int size, FlockConfig* config)
{
#pragma region Init
#ifdef USE_SHARED_MEM
	__shared__ Boid sharedBoids[THREADS_PER_BLOCK];
#endif

	const int tileSize = blockDim.x;
	const int tileCount = gridDim.x;
	const int boidId = blockDim.x * blockIdx.x + threadIdx.x;

#ifdef USE_SHARED_MEM
	float3 position = boids[min(boidId, size - 1)].position;
	float3 direction = vecNormalize(boids[min(boidId, size - 1)].direction);
#else
	if (boidId >= size) return;

	float3 position = boids[boidId].position;
	float3 direction = vecNormalize(boids[boidId].direction);
#endif

	Force force = { 0 };

#ifdef USE_SHARED_MEM
	int boidsLeft = size;
	for (int tile = 0; tile < tileCount - 1; tile++)
	{
		int tid = tile * tileSize + threadIdx.x;
		sharedBoids[threadIdx.x] = boids[tid];
		__syncthreads();

		for (int i = 0; i < tileSize; i++)
		{
			if (isInViewRange(position, direction, sharedBoids[i].position, config->viewAngle))
			{
				updateFlock(force, position, sharedBoids[i].position, sharedBoids[i].direction, config);
			}
		}
		boidsLeft -= tileSize;
		__syncthreads();
	}
	int tid = (tileCount - 1) * tileSize + threadIdx.x;
	if (tid < size)
	{
		sharedBoids[threadIdx.x] = boids[tid];
	}
	__syncthreads();

	for (int i = 0; i < boidsLeft; i++)
	{
		if (isInViewRange(position, direction, sharedBoids[i].position, config->viewAngle))
		{
			updateFlock(force, position, sharedBoids[i].position, sharedBoids[i].direction, config);
		}
	}
	__syncthreads();

	if (boidId >= size) return;
#else
	for (int i = 0; i < size; i++)
	{
		if (isInViewRange(position, direction, boids[i].position, config->viewAngle))
		{
			updateFlock(force, position, boids[i].position, boids[i].direction, config);
		}
	}
#endif

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

	Acceleration acc;
	acc.alignment = force.alignment * config->alignmentFactor;
	acc.separation = force.separation * config->separationFactor;

	if (force.cohesionCount > 0)
	{
		acc.cohesion = (force.cohesion - position) * config->cohesionFactor;
	}
	else acc.cohesion = make_float3(0.0f, 0.0f, 0.0f);
	
	acc.goal = vecNormalize(config->goal - position) * config->goalFactor;

	outAccelerations[boidId] = acc;
#pragma endregion
}
static __global__ void calculatePositions(Boid* boids, Acceleration* accelerations, size_t size, FlockConfig* config)
{
	const int boidId = blockDim.x * blockIdx.x + threadIdx.x;
	if (boidId >= size) return;

	float3 acc = accelerations[boidId].separation + accelerations[boidId].cohesion + accelerations[boidId].alignment + accelerations[boidId].goal;

	boids[boidId].direction += acc;
	boids[boidId].direction = vecClamp(boids[boidId].direction, config->maxVelocity);
	boids[boidId].position += boids[boidId].direction;
}

/// C++
static float getInitBoidRange()
{
	return log10(BOID_COUNT);
}
static std::vector<Boid> initBoids(int count)
{
	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_real_distribution<float> posDist(-getInitBoidRange(), getInitBoidRange());
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
static void copyTransformsFromCuda(DemoBoids* demo, CudaMemory<Boid>& boids, CudaMemory<Acceleration>& accelerations)
{
	std::vector<Boid> cpuBoids(BOID_COUNT);
	boids.load(*cpuBoids.data(), BOID_COUNT);

	std::vector<Acceleration> cpuAccelerations(BOID_COUNT);
	accelerations.load(*cpuAccelerations.data(), BOID_COUNT);

	SceneManager* manager = SceneManager::GetInstance();
	flockCenter = glm::vec3(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < cpuBoids.size(); i++)
	{
		demo->boids[i]->setTransforms(cpuBoids[i].position, cpuBoids[i].direction, cpuAccelerations[i]);
		demo->boids[i]->setViewAngle(boidsViewAngle);

		flockCenter += glm::vec3(cpuBoids[i].position.x, cpuBoids[i].position.y, cpuBoids[i].position.z);
	}

	flockCenter /= cpuBoids.size();
}

static FlockConfig update_config()
{
	FlockConfig config = { 0 };
	config.separationFactor = boidsSeparationFactor;
	config.cohesionFactor = boidsCohesionFactor;
	config.alignmentFactor = boidsAlignmentFactor;
	config.goalFactor = boidsGoalFactor;
	config.goal = make_float3(boidGoal.x, boidGoal.y, boidGoal.z);

	config.cohesionNeighbourhood = boidsCohesionNeighbourhood;
	config.separationNeighbourhood = boidsSeparationNeighbourhood;
	config.alignmentNeighbourhood = boidsAlignmentNeighbourhood;

	config.maxVelocity = boidsMaxVelocity;
	config.viewAngle = glm::radians(boidsViewAngle);

	return config;
}
static glm::vec3 getMousePos(Mouse* mouse)
{
	SceneData* sceneData = SceneManager::GetInstance()->m_sceneData;
	unsigned int* screen = SceneManager::GetInstance()->m_sceneSetting->m_screen;
	glm::vec3 position = glm::vec3(mouse->m_lastPosition[0], screen[1] - mouse->m_lastPosition[1], 1.0f);

	return glm::unProject(position, sceneData->cameras[0]->getVM(), sceneData->cameras[0]->getProjectionMatrix(), glm::vec4(0, 0, screen[0], screen[1]));
}
static void updateTarget(DemoBoids* demo)
{
	SceneData* sceneData = SceneManager::GetInstance()->m_sceneData;
	Mouse* mouse = sceneData->mouse;

	if (!mouse->clickPending) return;
	mouse->clickPending = false;

	glm::vec3 pos = getMousePos(mouse);
	glm::vec3 cameraPosition = sceneData->cameras[0]->getPosition();
	glm::vec3 toFlock = flockCenter - cameraPosition;
	glm::vec3 toTarget = glm::normalize(pos - cameraPosition);
	toTarget *= glm::dot(toFlock, toTarget);
	toTarget += cameraPosition;
		
	boidGoal = toTarget;
	demo->modelObjects[0]->setPosition(boidGoal.x, boidGoal.y, boidGoal.z);
}

static void boids_body(int argc, char** argv)
{
	srand((unsigned int) time(nullptr));

#ifdef VISUALIZE
	SceneManager* sceneManager = SceneManager::GetInstance();
	DemoBoids* demo = new DemoBoids(sceneManager->m_sceneData, BOID_COUNT);
	sceneManager->Init(argc, argv, demo);
	cudaGLSetGLDevice(0);
#endif

	std::vector<Boid> boids = initBoids(BOID_COUNT);
	CudaMemory<Boid> cudaBoids(boids.size(), boids.data());
	CudaMemory<Acceleration> cudaAccelerations(BOID_COUNT);

	dim3 blockDim(THREADS_PER_BLOCK, 1);
	dim3 gridDim(getNumberOfParts(BOID_COUNT, THREADS_PER_BLOCK), 1);

	FlockConfig flockConfig = update_config();
	CudaMemory<FlockConfig> flockConfigCuda(1, &flockConfig);

	while (true)
	{
#ifdef VISUALIZE
		flockConfigCuda.store(update_config());
#endif

#ifdef SIMULATE
		CudaTimer timer;
		timer.start();
		calculateAccelerations << <gridDim, blockDim >> > (cudaBoids.device(), cudaAccelerations.device(), BOID_COUNT, flockConfigCuda.device());
		timer.stop_wait();
#ifndef VISUALIZE
		timer.print("Update directions: ");
#endif
#endif

#ifdef SIMULATE
		timer.start();
		calculatePositions << <gridDim, blockDim >> > (cudaBoids.device(), cudaAccelerations.device(), BOID_COUNT, flockConfigCuda.device());
		timer.stop_wait();
		
#ifndef VISUALIZE
		timer.print("Update positions: ");
#endif
#endif

#ifdef VISUALIZE
		copyTransformsFromCuda(demo, cudaBoids, cudaAccelerations);
		updateTarget(demo);

		sceneManager->Refresh();

		Sleep(5);
#endif
	}
}
void boids(int argc, char** argv)
{
	boids_body(argc, argv);
}
