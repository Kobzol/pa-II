#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cudaDefs.h>

#include "cudamem.h"

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
		cudaEventCreate(&this->startEvent);
		cudaEventCreate(&this->stopEvent);

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
			std::cerr << this->get_time() << std::endl;
		}

		cudaEventDestroy(this->startEvent);
		cudaEventDestroy(this->stopEvent);
	}

	CudaTimer(const CudaTimer& other) = delete;
	CudaTimer& operator=(const CudaTimer& other) = delete;
	CudaTimer(CudaTimer&& other) = delete;

	void start() const
	{
		cudaEventRecord(this->startEvent);
	}
	void stop_wait() const
	{
		cudaEventRecord(this->stopEvent);
		cudaEventSynchronize(this->stopEvent);
	}
	float get_time() const
	{
		float time;
		cudaEventElapsedTime(&time, this->startEvent, this->stopEvent);
		return time;
	}

private:
	cudaEvent_t startEvent, stopEvent;
	bool automatic;
};
