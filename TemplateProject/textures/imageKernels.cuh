#ifndef __IMAGE_KERNEL_CUH_
#define __IMAGE_KERNEL_CUH_

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Converts color a float [pitch] linear memory - PREPARED FOR 2D CUDA BLOCK in generall,
/// 			but blockDim.y can be 1 as well.</summary>
/// <remarks>	18. 3. 2013. </remarks>
/// <typeparam name="unsigned int srcBPP"> Source bits per pixel. </param>
/// <param name="src">   	Source data. </param>
/// <param name="srcWidth"> The width. </param>
/// <param name="srcHeight">The height. </param>
/// <param name="srcHeight">The pitch of src (in number of elements). </param>
/// <param name="dstPitch">	The pitch of dst (in number of elements). </param>
/// <param name="dst">   	[in,out] If non-null, destination for the. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template<unsigned int srcBPP>
__global__ void colorToFloat(const unsigned char *src, const unsigned int srcWidth, const unsigned int srcHeight, const unsigned int srcPitch, const unsigned int dstPitch, float* dst )
{
	unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((tx < srcWidth) && (ty < srcHeight))
	{
		unsigned int dstOffset = ty * dstPitch + tx;
		printf("%d\n", dstOffset);
		unsigned int srcOffset = (ty * srcPitch + tx) * srcBPP/8;
		unsigned int value = 0;

		if (srcBPP>=8)
		{
			value = src[srcOffset++];
		}
		if (srcBPP>=16)
		{
			value = (value<<8) | src[srcOffset++];
		}
		if (srcBPP>=24)
		{
			value = (value<<8) | src[srcOffset++];
		}
		if (srcBPP>=32)
		{
			value = (value<<8) | src[srcOffset++];
		}
		dst[dstOffset] = (float)value;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Converts color a float [pitch] linear memory - PREPARED FOR 2D CUDA BLOCK in generall,
/// 			but blockDim.y can be 1 as well.</summary>
/// <remarks>	18. 3. 2013. </remarks>
/// <typeparam name="unsigned char srcBPP"> Source bits per pixel. </param>
///  <typeparam name="unsigned char blockDimension"> Block dimension = 1 for 1D or 2 for 2D </param>
/// <param name="src">   	Source data. </param>
/// <param name="srcWidth"> The width. </param>
/// <param name="srcHeight">The height. </param>
/// <param name="srcHeight">The pitch of src. </param>
/// <param name="dstPitch">	The pitch of dst. </param>
/// <param name="dst">   	[in,out] If non-null, destination for the. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template<unsigned char srcBPP, unsigned char blockDimension>
__global__ void colorToFloat(const unsigned char *src, const unsigned int srcWidth, const unsigned int srcHeight, const unsigned int srcPitch, const unsigned int dstPitch, float* dst)
{
	unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ty = (blockDimension == 2) ? blockIdx.y * blockDim.y + threadIdx.y : (blockIdx.x * blockDim.x + threadIdx.x) / srcWidth;
	if (blockDimension == 1)
	{
		tx -= ty * srcWidth;
	}

	if ((tx < srcWidth) && (ty < srcHeight))
	{
		unsigned int dstOffset = ty * dstPitch + tx;				//Consider dst as a ROW MATRIX !!!
		unsigned int srcOffset = (ty * srcPitch + tx) * srcBPP / 8;	//Consider src as a ROW MATRIX !!!
		unsigned int value = 0;

		if (srcBPP >= 8)
		{
			value = src[srcOffset++];
		}
		if (srcBPP >= 16)
		{
			value = (value << 8) | src[srcOffset++];
		}
		if (srcBPP >= 24)
		{
			value = (value << 8) | src[srcOffset++];
		}
		if (srcBPP >= 32)
		{
			value = (value << 8) | src[srcOffset++];
		}
		dst[dstOffset] = (float)value;
	}
}



#endif
