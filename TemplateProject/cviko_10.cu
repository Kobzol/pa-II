#include <cudaDefs.h>
#include <time.h>
#include <math.h>

static const unsigned int N = 1 << 20;
static const unsigned int MEMSIZE = N * sizeof(unsigned int);
static const unsigned int NO_LOOPS = 100;
static const unsigned int THREAD_PER_BLOCK = 256;
static const unsigned int GRID_SIZE = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

static void fillData(unsigned int *data, const unsigned int length)
{
	//srand(time(0));
	for (unsigned int i = 0; i<length; i++)
	{
		//data[i]= rand();
		data[i] = 1;
	}
}

static void printData(const unsigned int *data, const unsigned int length)
{
	if (data == 0) return;
	for (unsigned int i = 0; i<length; i++)
	{
		printf("%u ", data[i]);
	}
}


static __global__ void kernel(const unsigned int *a, const unsigned int *b, const unsigned int length, unsigned int *c)
{
	//TODO: Vector ADD 
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= length) return;
	c[tid] = a[tid] + b[tid];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 1. - single stream, async calling </summary>
///
/// <remarks>	16. 4. 2013. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////
static void test1()
{
	unsigned int *a, *b, *c;
	unsigned int *da, *db, *dc;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc((void**)&da, MEMSIZE);
	cudaMalloc((void**)&db, MEMSIZE);
	cudaMalloc((void**)&dc, MEMSIZE);

	//TODO: create stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	dim3 block(THREAD_PER_BLOCK);
	dim3 grid(getNumberOfParts(N, THREAD_PER_BLOCK));

	unsigned int dataOffset = 0;
	for (int i = 0; i < NO_LOOPS; i++)
	{
		//TODO:  copy a->da, b->db
		cudaMemcpyAsync(da, a + dataOffset, MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(db, b + dataOffset, MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);
		//TODO:  run the kernel in the stream
		kernel << <grid, block, 0, stream >> > (da, db, N, dc);
		//TODO:  copy dc->c
		cudaMemcpyAsync(c + dataOffset, dc, MEMSIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
		dataOffset += N;
	}

	//TODO: Synchonize stream
	cudaStreamSynchronize(stream);

	//TODO: Destroy stream
	cudaStreamDestroy(stream);

	printData(c, 100);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 2. - two streams - depth first approach </summary>
///
/// <remarks>	16. 4. 2013. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////
static void test2()
{
	//TODO: reuse the source code of above mentioned method test1()
	unsigned int *a, *b, *c;
	unsigned int *da[2], *db[2], *dc[2];

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	for (int i = 0; i < 2; i++)
	{
		cudaMalloc((void**)&da[i], MEMSIZE);
		cudaMalloc((void**)&db[i], MEMSIZE);
		cudaMalloc((void**)&dc[i], MEMSIZE);
	}

	//TODO: create stream
	cudaStream_t stream[2];
	for (int i = 0; i < 2; i++)
	{
		cudaStreamCreate(&stream[i]);
	}

	dim3 block(THREAD_PER_BLOCK);
	dim3 grid(getNumberOfParts(N, THREAD_PER_BLOCK));

	unsigned int dataOffset = 0;
	for (int i = 0; i < NO_LOOPS; i += 2)
	{
		//TODO:  copy a->da, b->db
		for (int s = 0; s < 2; s++)
		{
			cudaMemcpyAsync(da[s], a + dataOffset, MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream[s]);
			cudaMemcpyAsync(db[s], b + dataOffset, MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream[s]);
			//TODO:  run the kernel in the stream
			kernel << <grid, block, 0, stream[s] >> > (da[s], db[s], N, dc[s]);
			//TODO:  copy dc->c
			cudaMemcpyAsync(c + dataOffset, dc[s], MEMSIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream[s]);
			dataOffset += N;
		}
	}

	//TODO: Synchonize stream
	for (int i = 0; i < 2; i++)
	{
		cudaStreamSynchronize(stream[i]);
	}

	//TODO: Destroy stream
	for (int i = 0; i < 2; i++)
	{
		cudaStreamDestroy(stream[i]);
	}

	printData(c, 100);

	for (int i = 0; i < 2; i++)
	{
		cudaFree(da[i]);
		cudaFree(db[i]);
		cudaFree(dc[i]);
	}

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 3. - two streams - breadth first approach</summary>
///
/// <remarks>	Gajdi, 16. 4. 2013. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////
static void test3()
{
	//TODO: reuse the source code of above mentioned method test1()
	unsigned int *a, *b, *c;
	unsigned int *da[2], *db[2], *dc[2];

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	for (int i = 0; i < 2; i++)
	{
		cudaMalloc((void**)&da[i], MEMSIZE);
		cudaMalloc((void**)&db[i], MEMSIZE);
		cudaMalloc((void**)&dc[i], MEMSIZE);
	}

	//TODO: create stream
	cudaStream_t stream[2];
	for (int i = 0; i < 2; i++)
	{
		cudaStreamCreate(&stream[i]);
	}

	dim3 block(THREAD_PER_BLOCK);
	dim3 grid(getNumberOfParts(N, THREAD_PER_BLOCK));

	unsigned int dataOffset[2] = { 0, N };
	for (int i = 0; i < NO_LOOPS; i += 2)
	{
		//TODO:  copy a->da, b->db
		for (int s = 0; s < 2; s++)
		{
			cudaMemcpyAsync(da[s], a + dataOffset[s], MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream[s]);
		}
		for (int s = 0; s < 2; s++)
		{
			cudaMemcpyAsync(db[s], b + dataOffset[s], MEMSIZE, cudaMemcpyKind::cudaMemcpyHostToDevice, stream[s]);
		}
		for (int s = 0; s < 2; s++)
		{
			//TODO:  run the kernel in the stream
			kernel << <grid, block, 0, stream[s] >> > (da[s], db[s], N, dc[s]);
		}
		for (int s = 0; s < 2; s++)
		{
			//TODO:  copy dc->c
			cudaMemcpyAsync(c + dataOffset[s], dc[s], MEMSIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream[s]);
		}
		for (int s = 0; s < 2; s++)
		{
			dataOffset[s] += 2 * N;
		}
	}

	//TODO: Synchonize stream
	for (int i = 0; i < 2; i++)
	{
		cudaStreamSynchronize(stream[i]);
	}

	//TODO: Destroy stream
	for (int i = 0; i < 2; i++)
	{
		cudaStreamDestroy(stream[i]);
	}

	printData(c, 100);

	for (int i = 0; i < 2; i++)
	{
		cudaFree(da[i]);
		cudaFree(db[i]);
		cudaFree(dc[i]);
	}

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}


void cviko10()
{
	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;

	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	cudaEventRecord(startEvent, 0);
	test1();
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Test time: %f ms\n", elapsedTime);

	cudaEventRecord(startEvent, 0);
	test2();
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Test time: %f ms\n", elapsedTime);

	cudaEventRecord(startEvent, 0);
	test3();
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Test time: %f ms\n", elapsedTime);

	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);
}
