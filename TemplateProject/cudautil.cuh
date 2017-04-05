#pragma once

#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cudaDefs.h>

void checkCudaCall(cudaError_t error, const char* expression, const char* file, int line);
#define CHECK_CUDA_CALL(err) (checkCudaCall(err, #err, __FILE__, __LINE__))

#ifdef __INTELLISENSE__
	void __syncthreads();
#endif

#define CUDA_METHOD __device__ __host__

__device__ int getGlobalIdx_1D_1D();
__device__ int getGlobalIdx_1D_2D();
__device__ int getGlobalIdx_2D_1D();
__device__ int getGlobalIdx_2D_2D();

class CudaTimer
{
public:
	CudaTimer(bool automatic = false) : automatic(automatic)
	{
		CHECK_CUDA_CALL(cudaEventCreate(&this->startEvent));
		CHECK_CUDA_CALL(cudaEventCreate(&this->stopEvent));

		if (automatic)
		{
			this->start();
		}
	}
	~CudaTimer()
	{
		if (this->automatic)
		{
			this->stop_wait();
			std::cerr << this->get_time() << " ms" << std::endl;
		}

		CHECK_CUDA_CALL(cudaEventDestroy(this->startEvent));
		CHECK_CUDA_CALL(cudaEventDestroy(this->stopEvent));
	}

	CudaTimer(const CudaTimer& other) = delete;
	CudaTimer& operator=(const CudaTimer& other) = delete;
	CudaTimer(CudaTimer&& other) = delete;

	void start() const
	{
		CHECK_CUDA_CALL(cudaEventRecord(this->startEvent));
	}
	void stop_wait() const
	{
		CHECK_CUDA_CALL(cudaEventRecord(this->stopEvent));
		CHECK_CUDA_CALL(cudaEventSynchronize(this->stopEvent));
	}
	float get_time() const
	{
		float time;
		CHECK_CUDA_CALL(cudaEventElapsedTime(&time, this->startEvent, this->stopEvent));
		return time;
	}

	void print(const std::string& message)
	{
		std::cerr << message << this->get_time() << " ms" << std::endl;
	}

private:
	cudaEvent_t startEvent, stopEvent;
	bool automatic;
};
