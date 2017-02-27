#include <vector>
#include <memory>

#include "cudautil.cuh"

static __global__ void fillMatrix(int* __restrict__ matrix, const int pitch, const int width, const int height)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < width && row < height)
	{
		int index = row * pitch + col;
		matrix[index] = row * width + col;
	}
}

static void vectors()
{
	const int width = 10;
	const int height = 10;

	CudaPitchedMemory<int> pitchedMem(width, height);

	const dim3 blockSize(8, 8);
	const dim3 gridSize(getNumberOfParts(pitchedMem.get_pitch(), blockSize.x), getNumberOfParts(height, blockSize.y));

	fillMatrix << <gridSize, blockSize>> > (pitchedMem.device(), pitchedMem.get_pitch(), width, height);

	std::unique_ptr<int> result = std::make_unique<int>(width * height);
	pitchedMem.load(result.get());
}

void cviko2()
{
	srand((unsigned int) NULL);

	vectors();
}
