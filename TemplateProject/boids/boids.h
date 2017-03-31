#pragma once

// Simulation
// perching
// collision avoidance
// view angle check

// Graphics
// view angle visualization
// draw sprite or better model

// Performance
// unroll boid loop/make block size 32 and remove sync
// change uniforms to UBOs and pixel unpack buffers to transfer directly from CUDA

void boids(int argc, char** argv);
