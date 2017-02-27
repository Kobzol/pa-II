#include <cudaDefs.h>
#include <chrono>
#include <iostream>
#include <string>

#include "graphics/opengl.h"

void cviko1();
void cviko2();
void cviko3();

cudaDeviceProp deviceProp;

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	cviko3();

	getchar();

	return 0;
}