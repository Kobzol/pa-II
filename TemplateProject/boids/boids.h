#pragma once

// Simulation
// perching
// collision avoidance

// Graphics
// draw sprite or better model

// Performance
// change uniforms to UBOs and pixel unpack buffers to transfer directly from CUDA

#include <vector_types.h>

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
	float separationFactor;
	float cohesionFactor;
	float alignmentFactor;
	float goalFactor;

	float separationNeighbourhood;
	float cohesionNeighbourhood;
	float alignmentNeighbourhood;
	float3 goal;

	float maxVelocity;
	float viewAngle;
};
struct Acceleration
{
	float3 separation;
	float3 cohesion;
	float3 alignment;
	float3 goal;
};


void boids(int argc, char** argv);
