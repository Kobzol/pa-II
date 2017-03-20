#include "cudautil.cuh"
#include "cudamem.h"

struct ConstTest
{
	int x;
	int y;
};

static __constant__ int constInt;
static __constant__ ConstTest constTest;
static __constant__ int constArray[2];

static __global__ void test()
{
	printf("Int: %d\n", constInt);
	printf("Structure: %d, %d\n", constTest.x, constTest.y);
	printf("Array: %d, %d\n", constArray[0], constArray[1]);
}

void cviko4()
{
	ConstTest local;
	local.x = 5;
	local.y = 6;
	int localArray[2] = { 5, 6 };

	CudaConstant<ConstTest>::toDevice(constTest, local);
	CudaConstant<int>::toDevice(constArray[0], localArray, 2);
	CudaConstant<int>::toDevice(constInt, 5);

	test << <1, 1 >> > ();
	cudaDeviceSynchronize();

	int x;
	int arrX[2];
	ConstTest xStr;

	CudaConstant<int>::fromDevice(constInt, &x);
	CudaConstant<int>::fromDevice(constArray[0], &arrX[0], 2);
	CudaConstant<ConstTest>::fromDevice(constTest, &xStr);
}
