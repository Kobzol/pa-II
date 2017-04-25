#include <cudaDefs.h>
#include <cublas_v2.h>
#include <iostream>

static cublasStatus_t status = cublasStatus_t();
static cublasHandle_t handle = cublasHandle_t();

#define N (5)
#define DIM (3)
static const unsigned int MEMSIZE = N * DIM * sizeof(float);
static const unsigned int THREAD_PER_BLOCK = 128;
static const unsigned int GRID_SIZE = (N * DIM + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

static void fillData(float *data, const unsigned int length, const unsigned int dim)
{
	unsigned int id = 0;
	for (unsigned int i = 0; i<length; i++)
	{
		for (unsigned int j = 0; j<dim; j++)
		{
			data[id++] = i & 255;   //=i%256
		}
	}
}

static void fillDataWithNumber(float *data, const unsigned int length, const unsigned int dim, const float number)
{
	unsigned int id = 0;
	for (unsigned int i = 0; i<length; i++)
	{
		for (unsigned int j = 0; j<dim; j++)
		{
			data[id++] = number;
		}
	}
}

static __global__ void kernelPowerTwo(const float *a, const float *b, const unsigned int length, float *a2, float *b2)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= length) return;

	a2[tid] = a[tid] * a[tid];
	b2[tid] = b[tid] * b[tid];
}

void cviko11()
{
	status = cublasCreate(&handle);

	float alpha, beta;
	float *a, *b, *m;
	float *da, *da2, *db, *db2, *dm;
	float *ones, *dones;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&ones, MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&m, N * N * sizeof(float), cudaHostAllocDefault);

	cudaMalloc((void**)&da, MEMSIZE);
	cudaMalloc((void**)&da2, MEMSIZE);
	cudaMalloc((void**)&db, MEMSIZE);
	cudaMalloc((void**)&db2, MEMSIZE);
	cudaMalloc((void**)&dones, MEMSIZE);
	cudaMalloc((void**)&dm, N * N * sizeof(float));

	fillData(a, N, DIM);
	fillData(b, N, DIM);
	fillDataWithNumber(ones, N, DIM, 1.0f);

	//Copy data to DEVICE
	cudaMemcpy(da, a, MEMSIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, MEMSIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dones, ones, MEMSIZE, cudaMemcpyHostToDevice);

	//TODO 1: Process a -> a^2  and b->b^2
	kernelPowerTwo << <GRID_SIZE, THREAD_PER_BLOCK >> >(da, db, N * DIM, da2, db2);

	//TODO 2: Process a^2 + b^2 using CUBLAS //pair-wise operation such that the result is dm[N*N] matrix
	alpha = 1.0f;
	beta = 0.0f;

	cublasSgemm(handle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N, N, N, DIM, &alpha, da2, DIM, dones, DIM, &beta, dm, N);
	beta = 1.0f;
	cublasSgemm(handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, N, N, DIM, &alpha, dones, N, db2, DIM, &beta, dm, N);

	//TODO 3: Process -2ab and sum with previous result stored in dm using CUBLAS
	alpha = -2.0f;
	cublasSgemm(handle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N, N, N, DIM, &alpha, da, DIM, db, DIM, &beta, dm, N);

	checkDeviceMatrix<float>(da, sizeof(float)*DIM, N, DIM, "%f ", "A");
	checkDeviceMatrix<float>(da2, sizeof(float)*DIM, N, DIM, "%f ", "A^2");
	checkDeviceMatrix<float>(db, sizeof(float)*DIM, N, DIM, "%f ", "B");
	checkDeviceMatrix<float>(db2, sizeof(float)*DIM, N, DIM, "%f ", "B^2");
	checkDeviceMatrix<float>(dones, sizeof(float)*DIM, N, DIM, "%f ", "ONES");
	checkDeviceMatrix<float>(dm, sizeof(float)*N, N, N, "%f ", "M");

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			int iIndex = i * 3;
			int jIndex = j * 3;
			float distSquared = std::pow(a[iIndex] - b[jIndex], 2) + std::pow(a[(iIndex + 1)] - b[(jIndex + 1)], 2) + std::pow(a[(iIndex + 2)] - b[(jIndex + 2)], 2);
			std::cerr << distSquared << " ";
		}
		std::cerr << std::endl;
	}

	cudaFree(da);
	cudaFree(da2);
	cudaFree(db);
	cudaFree(db2);
	cudaFree(dm);
	cudaFree(dones);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(m);
	cudaFreeHost(ones);

	status = cublasDestroy(handle);
}
