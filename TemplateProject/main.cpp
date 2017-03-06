#include <cudaDefs.h>
#include <chrono>
#include <iostream>
#include <string>

#include "boids/boids.h"

void cviko1();
void cviko2();
void cviko3();
void cviko4();

cudaDeviceProp deviceProp;

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	boids(argc, argv);

	getchar();

	return 0;
}
