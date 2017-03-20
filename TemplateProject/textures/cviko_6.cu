// includes, cudaimageWidth
#include <cudaDefs.h>
#include <imageManager.h>

#include "../cudautil.cuh"
#include "imageKernels.cuh"

#define BLOCK_DIM 8


//Use the followings to store information about the input image that will be processed
static unsigned char* dSrcImageData = nullptr;
static unsigned int srcImageWidth;
static unsigned int srcImageHeight;
static unsigned int srcImageBPP;		//Bits Per Pxel = 8, 16, 24, or 32 bit
static unsigned int srcImagePitch;

//Use the followings to access the input image through the texture reference
texture<float, 2, cudaReadModeElementType> srcTexRef;
static cudaChannelFormatDesc srcTexCFD;
static size_t srcTexPitch;
static float* dSrcTexData = nullptr;

cudaTextureObject_t textureObj;
cudaResourceDesc textureResDesc;
cudaTextureDesc textureDesc;
cudaResourceViewDesc textureViewDesc;

static size_t dstTexPitch;
static uchar3* dstTexData = nullptr;

static KernelSetting squareKs;
static float* dOutputData = nullptr;

template<bool normalizeTexel>
static __global__ void floatHeighmapTextureToNormalmap(cudaTextureObject_t tex, const unsigned int texWidth, const unsigned int texHeight, const unsigned int dstPitch, uchar3* dst)
{
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	if (tidX < texWidth && tidY < texHeight)
	{
		float texX = tidX;
		float texY = tidY;

		// sobel x
		float lu = tex2D<float>(tex, texX - 1, texY - 1) * -1.0f;
		float lc = tex2D<float>(tex, texX - 1, texY) * -2.0f;
		float lb = tex2D<float>(tex, texX - 1, texY + 1) * -1.0f;
		float ru = tex2D<float>(tex, texX + 1, texY - 1);
		float rc = tex2D<float>(tex, texX + 1, texY) * 2.0f;
		float rb = tex2D<float>(tex, texX + 1, texY + 1);

		float3 value = make_float3(0.0f, lu + lc + lb + ru + rc + rb, 0.0f);

		// sobel y
		lu = tex2D<float>(tex, texX - 1, texY - 1);				// left up
		lc = tex2D<float>(tex, texX, texY - 1) * 2.0f;			// center up
		lb = tex2D<float>(tex, texX + 1, texY - 1);				// right up
		ru = tex2D<float>(tex, texX - 1, texY + 1) * -1.0f;		// left bottom
		rc = tex2D<float>(tex, texX, texY + 1) * -2.0f;			// center bottom
		rb = tex2D<float>(tex, texX + 1, texY + 1) * -1.0f;		// right bottom

		value.z = lu + lc + lb + ru + rc + rb;

		char* mem = ((char*) dst) + (tidY * dstPitch + (tidX * sizeof(uchar3)));
		uchar3* dstMem = (uchar3*) mem;

		value += 1;
		value /= 2.0f;
		value.x = 0.5f;
		value *= 255.0f;

		*dstMem = make_uchar3(value.x, value.y, value.z);
	}
}

#pragma region STEP 1

//TASK:	Load the input image and store loaded data in DEVICE memory (dSrcImageData)

static void loadSourceImage(const char* imageFileName)
{
	FreeImage_Initialise();
	FIBITMAP *tmp = ImageManager::GenericLoader(imageFileName, 0);

	srcImageWidth = FreeImage_GetWidth(tmp);
	srcImageHeight = FreeImage_GetHeight(tmp);
	srcImageBPP = FreeImage_GetBPP(tmp);
	srcImagePitch = FreeImage_GetPitch(tmp);		// FREEIMAGE aligns row data ... You have to use pitch instead of width

	CHECK_CUDA_CALL(cudaMalloc((void**)&dSrcImageData, srcImagePitch * srcImageHeight * srcImageBPP/8));
	CHECK_CUDA_CALL(cudaMemcpy(dSrcImageData, FreeImage_GetBits(tmp), srcImagePitch * srcImageHeight * srcImageBPP/8, cudaMemcpyHostToDevice));

	//checkHostMatrix<unsigned char>(FreeImage_GetBits(tmp), srcImagePitch, srcImageHeight, srcImageWidth, "%hhu ", "Result of Linear Pitch Text");
	//checkDeviceMatrix<unsigned char>(dSrcImageData, srcImagePitch, srcImageHeight, srcImageWidth, "%hhu ", "Result of Linear Pitch Text");
	
	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}
#pragma endregion

#pragma region STEP 2

//TASK: Create a texture based on the source image. The input images can have variable BPP (Byte Per Pixel), but finally any such image will be converted into the floating-point texture using
//		the colorToFloat kernel.

static void createSrcTexure()
{
	//TODO: Floating Point Texture Data
	CHECK_CUDA_CALL(cudaMallocPitch(&dSrcTexData, &srcTexPitch, srcImageWidth * sizeof(float), srcImageHeight));

	//Converts custom image data to float and stores result in the float_pitch_linear_data
	switch(srcImageBPP)
	{
		case 8:  colorToFloat<8 , 2><<<squareKs.dimGrid, squareKs.dimBlock>>>(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, srcTexPitch/sizeof(float), dSrcTexData); break;
		case 16: colorToFloat<16, 2><<<squareKs.dimGrid, squareKs.dimBlock>>>(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, srcTexPitch/sizeof(float), dSrcTexData); break;
		case 24: colorToFloat<24, 2><<<squareKs.dimGrid, squareKs.dimBlock>>>(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, srcTexPitch/sizeof(float), dSrcTexData); break;
		case 32: colorToFloat<32, 2><<<squareKs.dimGrid, squareKs.dimBlock>>>(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, srcTexPitch/sizeof(float), dSrcTexData); break;
	}
	//checkDeviceMatrix<float>(dSrcTexData, srcTexPitch, srcImageHeight, srcImageWidth, "%6.1f ", "Result of Linear Pitch Text");

	//TODO: Texture settings
	srcTexCFD = cudaCreateChannelDesc<float>();
	
	srcTexRef.normalized = 0;
	srcTexRef.addressMode[0] = cudaAddressModeClamp;	// horizontal
	srcTexRef.addressMode[1] = cudaAddressModeClamp;	// vertical
	srcTexRef.filterMode = cudaFilterModePoint;

	textureResDesc.resType = cudaResourceTypePitch2D;
	textureResDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
	textureResDesc.res.pitch2D.devPtr = dSrcTexData;
	textureResDesc.res.pitch2D.width = srcImageWidth;
	textureResDesc.res.pitch2D.height = srcImageHeight;
	textureResDesc.res.pitch2D.pitchInBytes = srcTexPitch;

	textureDesc.addressMode[0] = cudaAddressModeClamp;
	textureDesc.addressMode[1] = cudaAddressModeClamp;
	textureDesc.normalizedCoords = 0;
	textureDesc.filterMode = cudaFilterModePoint;

	CHECK_CUDA_CALL(cudaCreateTextureObject(&textureObj, &textureResDesc, &textureDesc, nullptr));

	//TODO: Bind texture
	CHECK_CUDA_CALL(cudaBindTexture2D(NULL, &srcTexRef, dSrcTexData, &srcTexCFD, srcImageWidth, srcImageHeight, srcTexPitch));
}
#pragma endregion

#pragma region STEP 3

//TASK:	Convert the input image into normal map. Use the binded texture (srcTexRef).

static void createNormalMap()
{
	//TODO: Allocate Pitch memory dstTexData to store output texture
	CHECK_CUDA_CALL(cudaMallocPitch(&dstTexData, &dstTexPitch, srcImageWidth * sizeof(uchar3), srcImageHeight));

	{
		CudaTimer timer(true);
		floatHeighmapTextureToNormalmap<true> << <squareKs.dimGrid, squareKs.dimBlock >> >(textureObj, srcImageWidth, srcImageHeight, dstTexPitch, dstTexData);
	}

	//check_data<uchar3>::checkDeviceMatrix(dstTexData, srcImageHeight, dstTexPitch/sizeof(uchar3), true, "%hhu %hhu %hhu %hhu | ", "Result of Linear Pitch Text");
}

#pragma endregion

#pragma region STEP 4

//TASK: Save output image (normal map)

static void saveTexImage(const char* imageFileName)
{
	FreeImage_Initialise();

	FIBITMAP *tmp = FreeImage_Allocate(srcImageWidth, srcImageHeight, 24);
	unsigned int tmpPitch = srcImagePitch = FreeImage_GetPitch(tmp);		// FREEIMAGE align row data ... You have to use pitch instead of width
	CHECK_CUDA_CALL(cudaMemcpy2D(FreeImage_GetBits(tmp) , FreeImage_GetPitch(tmp), dstTexData, dstTexPitch, srcImageWidth * 3, srcImageHeight, cudaMemcpyDeviceToHost));
    FreeImage_Save(FIF_PNG, tmp, imageFileName, 0);
	ImageManager::GenericWriter(tmp, imageFileName, FIF_PNG);
    FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}

#pragma endregion

static void releaseMemory()
{
	CHECK_CUDA_CALL(cudaUnbindTexture(srcTexRef));
	if (dSrcImageData!=0)
		CHECK_CUDA_CALL(cudaFree(dSrcImageData));
	if (dSrcTexData!=0)
		CHECK_CUDA_CALL(cudaFree(dSrcTexData));
	if (dstTexData!=0)
		CHECK_CUDA_CALL(cudaFree(dstTexData));
	if (dOutputData)
		CHECK_CUDA_CALL(cudaFree(dOutputData));
}

void cviko6()
{
	//STEP 1
	loadSourceImage("textures/terrain3Kx3K.tif");

	//TODO: Setup the kernel settings
	squareKs.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	squareKs.blockSize = BLOCK_DIM * BLOCK_DIM;
	squareKs.dimGrid = dim3(getNumberOfParts(srcImageWidth, BLOCK_DIM), getNumberOfParts(srcImageHeight, BLOCK_DIM), 1);

	//Step 2 - create heighmap texture stored in the linear pitch memory
	createSrcTexure();

	//Step 3 - create the normal map
	createNormalMap();

	//Step 4 - save the normal map
	saveTexImage("textures/normalMap.png");

	releaseMemory();
}
