#include <cudaDefs.h>
#include <chrono>
#include <iostream>
#include <string>

#include "boids/boids.h"

void cviko1();
void cviko2();
void cviko3();
void cviko4();
void cviko5();
void cviko6();
void cviko7(int argc, char** argv);

cudaDeviceProp deviceProp;

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	cviko7(argc, argv);

	getchar();

	return 0;
}
