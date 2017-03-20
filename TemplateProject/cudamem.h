#pragma once

#include <iostream>
#include <cassert>
#include "cudautil.cuh"

/*
 * Represents RAII CUDA memory allocated on (GPU) device.
 */
template <typename T>
class CudaMemory
{
public:
	CudaMemory(size_t count = 1, T* mem = nullptr) : count(count)
	{
		CHECK_CUDA_CALL(cudaMalloc(&this->devicePointer, sizeof(T) * count));

		if (mem)
		{
			this->store(*mem, count);
		}
	}
	CudaMemory(size_t count, T value) : count(count)
	{
		CHECK_CUDA_CALL(cudaMalloc(&this->devicePointer, sizeof(T) * count));
		CHECK_CUDA_CALL(cudaMemset(this->devicePointer, value, sizeof(T) * count));
	}
	~CudaMemory()
	{
		CHECK_CUDA_CALL(cudaFree(this->devicePointer));
		this->devicePointer = nullptr;
	}

	CudaMemory(const CudaMemory& other) = delete;
	CudaMemory& operator=(const CudaMemory& other) = delete;
	CudaMemory(CudaMemory&& other) = delete;

	T* operator*() const
	{
		return this->devicePointer;
	}
	T* device() const
	{
		return this->devicePointer;
	}

	void load(T& dest, size_t count = 1) const
	{
		if (count == 0)
		{
			count = this->count;
		}

		CHECK_CUDA_CALL(cudaMemcpy(&dest, this->devicePointer, sizeof(T) * count, cudaMemcpyDeviceToHost));
	}
	void store(T& src, size_t count = 1, size_t start_index = 0)
	{
		if (count == 0)
		{
			count = this->count;
		}

		CHECK_CUDA_CALL(cudaMemcpy(this->devicePointer + start_index, &src, sizeof(T) * count, cudaMemcpyHostToDevice));
	}

private:
	T* devicePointer = nullptr;
	size_t count;
};

/*
* Represents RAII pitched CUDA memory allocated on (GPU) device.
*/
template <typename T>
class CudaPitchedMemory
{
public:
	CudaPitchedMemory(size_t width, size_t height) : width(width), height(height)
	{
		CHECK_CUDA_CALL(cudaMallocPitch(&this->devicePointer, &this->pitch, width, height));
	}
	~CudaPitchedMemory()
	{
		CHECK_CUDA_CALL(cudaFree(this->devicePointer));
		this->devicePointer = nullptr;
	}

	CudaPitchedMemory(const CudaPitchedMemory& other) = delete;
	CudaPitchedMemory& operator=(const CudaPitchedMemory& other) = delete;
	CudaPitchedMemory(CudaPitchedMemory&& other) = delete;

	T* operator*() const
	{
		return this->devicePointer;
	}
	T* device() const
	{
		return this->devicePointer;
	}

	void load(T* dest) const
	{
		CHECK_CUDA_CALL(cudaMemcpy2D(dest, sizeof(T) * this->width, this->devicePointer, this->get_pitch_bytes(), sizeof(T) * this->width, this->height, cudaMemcpyDeviceToHost));
	}

	size_t get_pitch() const
	{
		return this->get_pitch_bytes() / sizeof(T);
	}
	size_t get_pitch_bytes() const
	{
		return this->pitch;
	}

private:
	T* devicePointer = nullptr;
	size_t width;
	size_t height;
	size_t pitch;
};

/*
 * Represents RAII CUDA pinned memory allocated on (CPU) host, directly mapped and accessible from device.
 * Requires 64-bit CUDA context and cudaDeviceMapHost flag to be set.
 */
template <typename T>
class CudaHostMemory
{
public:
	CudaHostMemory(size_t count = 1) : count(count)
	{
		cudaMallocHost(&hostPointer, sizeof(T) * count);
		cudaHostGetDevicePointer(&this->devicePointer, this->hostPointer, 0);
	}
	~CudaHostMemory()
	{
		CHECK_CUDA_CALL(cudaFreeHost(this->hostPointer));
		this->hostPointer = nullptr;
		this->devicePointer = nullptr;
	}

	T* host() const
	{
		return this->hostPointer;
	}
	T* device() const
	{
		return this->devicePointer;
	}

private:
	T* hostPointer;
	T* devicePointer;
	size_t count;
};

/*
* Represents constant CUDA memory.
*/
template <typename T>
class CudaConstant
{
public:
	static void toDevice(const T& symbol, const T& source, size_t size = 1)
	{
		CHECK_CUDA_CALL(cudaMemcpyToSymbol(symbol, &source, size * sizeof(T)));
	}
	static void toDevice(const T& symbol, const T* source, size_t size = 1)
	{
		CHECK_CUDA_CALL(cudaMemcpyToSymbol(symbol, source, size * sizeof(T)));
	}
	static void fromDevice(const T& symbol, T* dest, size_t size = 1)
	{
		CHECK_CUDA_CALL(cudaMemcpyFromSymbol(dest, symbol, size * sizeof(T)));
	}
};
