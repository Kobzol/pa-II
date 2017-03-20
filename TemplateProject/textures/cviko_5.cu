// includes, cuda
#include <cuda_runtime.h>

#include <cudaDefs.h>
#include <Utils/imageManager.h>
#include "../cudautil.cuh"

#include "imageKernels.cuh"

#define BLOCK_DIM 8

static cudaError_t error = cudaSuccess;
static cudaDeviceProp deviceProp = cudaDeviceProp();

texture<float, 2, cudaReadModeElementType> texRef;		// declared texture reference must be at file-scope !!!

static cudaChannelFormatDesc texChannelDesc;

static unsigned char* dImageData = nullptr;
static unsigned int imageWidth;
static unsigned int imageHeight;
static unsigned int imageBPP;		//Bits Per Pixel = 8, 16, 24, or 32 bit
static unsigned int imagePitch;

static size_t texPitch;
static float* dLinearPitchTextureData = nullptr;
static cudaArray* dArrayTextureData = nullptr;

static KernelSetting ks;

static float* dOutputData = nullptr;

static void loadSourceImage(const char* imageFileName)
{
	FreeImage_Initialise();
	FIBITMAP* tmp = ImageManager::GenericLoader(imageFileName, 0);

	imageWidth = FreeImage_GetWidth(tmp);
	imageHeight = FreeImage_GetHeight(tmp);
	imageBPP = FreeImage_GetBPP(tmp);
	imagePitch = FreeImage_GetPitch(tmp);		// FREEIMAGE align row data ... You have to use pitch instead of width

	CHECK_CUDA_CALL(cudaMalloc((void**)&dImageData, imagePitch * imageHeight * imageBPP/8));
	CHECK_CUDA_CALL(cudaMemcpy(dImageData, FreeImage_GetBits(tmp), imagePitch * imageHeight * imageBPP/8, cudaMemcpyHostToDevice));

	checkHostMatrix<unsigned char>(FreeImage_GetBits(tmp), imagePitch, imageHeight, imageWidth, "%hhu ", "Result of Linear Pitch Text");
	checkDeviceMatrix<unsigned char>(dImageData, imagePitch, imageHeight, imageWidth, "%hhu ", "Result of Linear Pitch Text");

	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}

static void createTextureFromLinearPitchMemory()
{
	// TODO: Allocate dLinearPitchTextureData variable memory
	CHECK_CUDA_CALL(cudaMallocPitch(&dLinearPitchTextureData, &texPitch, imageWidth * sizeof(float), imageHeight));

	dim3 blockDim(8, 8);
	dim3 gridDim(getNumberOfParts(imageWidth, blockDim.x), getNumberOfParts(imageHeight, blockDim.y));

	switch (imageBPP)
	{
		//TODO: Here call your kernel to convert image into linearPitch memory
	case 8:
		colorToFloat<8> << <gridDim, blockDim>> > (dImageData, imageWidth, imageHeight, imagePitch, texPitch / sizeof(float), dLinearPitchTextureData);
		break;
	case 16:
		colorToFloat<16> << <gridDim, blockDim >> > (dImageData, imageWidth, imageHeight, imagePitch, texPitch / sizeof(float), dLinearPitchTextureData);
		break;
	case 24:
		colorToFloat<24> << <gridDim, blockDim >> > (dImageData, imageWidth, imageHeight, imagePitch, texPitch / sizeof(float), dLinearPitchTextureData);
		break;
	case 32:
		colorToFloat<32> << <gridDim, blockDim >> > (dImageData, imageWidth, imageHeight, imagePitch, texPitch / sizeof(float), dLinearPitchTextureData);
		break;
	}

	checkDeviceMatrix<float>(dLinearPitchTextureData, texPitch, imageHeight, imageWidth, "%6.1f ", "Result of Linear Pitch Text");

	//TODO: Define texture channel descriptor (texChannelDesc)
	texChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); //cudaCreateChannelDesc<float>();

	//TODO: Define texture (texRef) parameters
	texRef.normalized = 0;
	texRef.addressMode[0] = cudaAddressModeClamp;	// horizontal
	texRef.addressMode[1] = cudaAddressModeClamp;	// vertical
	texRef.filterMode = cudaFilterModePoint;

	//TODO: Bind texture
	CHECK_CUDA_CALL(cudaBindTexture2D(NULL, &texRef, dLinearPitchTextureData, &texChannelDesc, imageWidth, imageHeight, texPitch));
}
static void createTextureFrom2DArray()
{
	//TODO: Define texture channel descriptor (texChannelDesc)
	texChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); //cudaCreateChannelDesc<float>();
						
	//TODO: Define texture (texRef) parameters
	texRef.normalized = 0;
	texRef.addressMode[0] = cudaAddressModeClamp;	// horizontal
	texRef.addressMode[1] = cudaAddressModeClamp;	// vertical
	texRef.filterMode = cudaFilterModePoint;

	//Converts custom image data to float and stores result in the float_linear_data
	float* dLinearTextureData = nullptr;
	CHECK_CUDA_CALL(cudaMalloc((void**)&dLinearTextureData, imageWidth * imageHeight * sizeof(float)));

	dim3 blockDim(8, 8);
	dim3 gridDim(getNumberOfParts(imageWidth, blockDim.x), getNumberOfParts(imageHeight, blockDim.y));

	switch(imageBPP)
	{
		//TODO: Here call your kernel to convert image into linear memory (no pitch!!!)
	case 8:
		colorToFloat<8> << <gridDim, blockDim >> > (dImageData, imageWidth, imageHeight, imagePitch, imageWidth, dLinearTextureData);
		break;
	case 16:
		colorToFloat<16> << <gridDim, blockDim >> > (dImageData, imageWidth, imageHeight, imagePitch, imageWidth, dLinearTextureData);
		break;
	case 24:
		colorToFloat<24> << <gridDim, blockDim >> > (dImageData, imageWidth, imageHeight, imagePitch, imageWidth, dLinearTextureData);
		break;
	case 32:
		colorToFloat<32> << <gridDim, blockDim >> > (dImageData, imageWidth, imageHeight, imagePitch, imageWidth, dLinearTextureData);
		break;
	}

	checkDeviceMatrix<float>(dLinearTextureData, imageWidth, imageHeight, imageWidth, "%6.1f ", "Result of Linear Text");

	CHECK_CUDA_CALL(cudaMallocArray(&dArrayTextureData, &texChannelDesc, imageWidth, imageHeight));
	
	//TODO: copy data into cuda array (dArrayTextureData)
	CHECK_CUDA_CALL(cudaMemcpyToArray(dArrayTextureData, 0, 0, dLinearTextureData, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToDevice));
	
	checkDeviceArray<float>(dArrayTextureData, imageWidth, imageHeight, imageWidth, "%6.1f", "Texture array");

	//TODO: Bind texture
	CHECK_CUDA_CALL(cudaBindTextureToArray(&texRef, dArrayTextureData, &texChannelDesc));
	CHECK_CUDA_CALL(cudaFree(dLinearTextureData));
}

static void releaseMemory()
{
	CHECK_CUDA_CALL(cudaUnbindTexture(&texRef));
	if (dImageData!=0)
		CHECK_CUDA_CALL(cudaFree(dImageData));
	if (dLinearPitchTextureData!=0)
		CHECK_CUDA_CALL(cudaFree(dLinearPitchTextureData));
	if (dArrayTextureData)
		CHECK_CUDA_CALL(cudaFreeArray(dArrayTextureData));
	if (dOutputData)
		CHECK_CUDA_CALL(cudaFree(dOutputData));
}

static __global__ void texKernel(const unsigned int texWidth, const unsigned int texHeight, float* dst)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	//TODO some kernel
}

void cviko5()
{
	loadSourceImage("textures/terrain10x10.tif");

	CHECK_CUDA_CALL(cudaMalloc((void**)&dOutputData, imageWidth * imageHeight * sizeof(float)));

	ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	ks.blockSize = BLOCK_DIM * BLOCK_DIM;
	ks.dimGrid = dim3((imageWidth + BLOCK_DIM-1)/BLOCK_DIM, (imageHeight + BLOCK_DIM-1)/BLOCK_DIM, 1);

	//Test 1 - texture stored in linear pitch memory
	/*createTextureFromLinearPitchMemory();
	texKernel<<<ks.dimGrid, ks.dimBlock>>>(imageWidth, imageHeight, dOutputData);
	checkDeviceMatrix<float>(dOutputData, imageWidth * sizeof(float), imageHeight, imageWidth, "%6.1f ", "dOutputData");*/

	//Test 2 - texture stored in 2D array
	createTextureFrom2DArray();
	texKernel<<<ks.dimGrid, ks.dimBlock>>>(imageWidth, imageHeight, dOutputData);
	checkDeviceMatrix<float>(dOutputData,  imageWidth * sizeof(float), imageHeight, imageWidth, "%6.1f ", "dOutputData");

	releaseMemory();
}
