#include <cudaDefs.h>
#include <chrono>
#include <iostream>
#include <string>

void cviko1();
void cviko2();
void cviko3();
void cviko4();
void cviko5();
void cviko6();
void cviko7(int argc, char** argv);
void cviko10();
void cviko11();

void boids(int argc, char** argv);

static cudaDeviceProp deviceProp;

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	boids(argc, argv);
	//cviko11();

	//getchar();

	return 0;
}
