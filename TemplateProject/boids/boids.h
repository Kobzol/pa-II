#pragma once

// Simulation
// perching
// collision avoidance
// view angle

// Graphics
// boid rotation
// view angle visualization
// draw sprites or better models

// Performance
// unroll boid loop/make block size 32 and remove sync
// change uniforms to UBOs and pixel unpack buffers to transfer directly from CUDA

void boids(int argc, char** argv);
