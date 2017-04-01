////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	helperBuffer.h
//
// summary:	Declaration and implementation of Helper methods
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __HELPERBUFFER_H_
#define __HELPERBUFFER_H_

#include <glew.h>
#include <FreeImage.h>
#include <buffer.h>

static inline void* getMappedPointer(const GLenum target, const GLuint buffer)
{
	int currentBufferSize = 0;
	glBindBuffer(target, buffer);
	glGetBufferParameteriv(target, GL_BUFFER_SIZE, &currentBufferSize);
	return (void*)glMapBufferRange(target, 0, currentBufferSize, GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
}

static inline void releaseMappedPointer(const GLenum target)
{
	if (!glUnmapBuffer(target))
	{
		printf("Unmap buffer failed.\n");
		return;
	}
	glBindBuffer(target, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	We do not use glBufferData for data upload, since it would load into host ram as pinned memory instead of loading into GPU RAM. 
/// 			With glMapBuffer+memcpy we ensure to have our data located on GPU. </summary>
/// <remarks>	Gajdi, 28. 12. 2013. </remarks>
/// <param name="buffer">   	The buffer. </param>
/// <param name="host_data">	The Pointer to the data stored on HOST (ram). </param>
/// <param name="sizeInBytes">	Size in bytes ... the number of elements to be coppied * size of the element. </param>
/// <param name="target">   	Target for the buffer [e.g. GL_ARRAY_BUFFER]. </param>
/// <param name="access">   	The access to the buffer [e.g. GL_STATIC_DRAW]. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
static inline void initBuffer(const GLuint buffer, const void* host_data, const unsigned sizeInBytes, const GLenum target, const GLenum access)
{
	glBindBuffer(target, buffer);
	glBufferData(target, sizeInBytes, NULL, access);

	if (host_data)
	{
		void* device_data = (void*)glMapBuffer(target, GL_WRITE_ONLY);
		if (device_data == NULL)
		{
			printf("Can not map gpu buffer.\n");
			return;
		}
		memcpy(device_data, host_data, sizeInBytes);					//Copy HOST->DEVICE
		if (!glUnmapBuffer(target))
		{
			printf("Unmap buffer failed.\n");
			return;
		}
		device_data = NULL;
	}
	glBindBuffer(target, 0);										// release buffer
}

static inline void updateBuffer(const GLuint buffer, const void* host_data, const unsigned sizeInBytes, const GLenum target, const GLenum access)
{
	int currentBufferSize = 0;
	glBindBuffer(target, buffer);
	glGetBufferParameteriv(target, GL_BUFFER_SIZE, &currentBufferSize);
	if (currentBufferSize != sizeInBytes)
	{
		initBuffer(buffer, host_data, sizeInBytes, target, access);
		return;
	}
	//void* device_data = (void*) glMapBuffer(target, GL_WRITE_ONLY);
	void* device_data = (void*)glMapBufferRange(target, 0, currentBufferSize, GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
	if (device_data == NULL)
	{
		printf("Can not map gpu buffer.\n");
		return;
	}
	memcpy(device_data, host_data, sizeInBytes);			//Copy HOST->DEVICE
	if (!glUnmapBuffer(target))
	{
		printf("Unmap buffer failed.\n");
		return;
	}
	glBindBuffer(target, 0);

}

static inline void getBufferSize(const GLint target, const GLenum attachment, GLint &width, GLint &height)
{
	GLint srcType = 0;
	GLint srcTex = 0;
	GLint srcLevel = 0;
	GLint srcWidth = 0;
	GLint srcHeight = 0;
	GLint srcFormat = 0;
	GLint srcCubeFace = 0;

	glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, attachment, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &srcType);					//GL_TEXTURE = 5890 = 0x1702
	glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, attachment, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &srcTex);
	glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, attachment, GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL, &srcLevel);
	glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, attachment, GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE, &srcCubeFace);

	switch (srcType)
	{
	case GL_RENDERBUFFER:
		glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &srcWidth);
		glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &srcHeight);
		break;
	case GL_TEXTURE:
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(target, srcTex);

		if (srcCubeFace == 0)
		{
			glGetTexLevelParameteriv(target, srcLevel, GL_TEXTURE_WIDTH, &srcWidth);
			glGetTexLevelParameteriv(target, srcLevel, GL_TEXTURE_HEIGHT, &srcHeight);
			glGetTexLevelParameteriv(target, srcLevel, GL_TEXTURE_INTERNAL_FORMAT, &srcFormat); // get internal format type of GL texture, e.g. GL_RGBA32F
		}
		else
		{
			glGetTexLevelParameteriv(srcCubeFace, srcLevel, GL_TEXTURE_WIDTH, &srcWidth);
			glGetTexLevelParameteriv(srcCubeFace, srcLevel, GL_TEXTURE_HEIGHT, &srcHeight);
			glGetTexLevelParameteriv(srcCubeFace, srcLevel, GL_TEXTURE_INTERNAL_FORMAT, &srcFormat); // get internal format type of GL texture, e.g. GL_RGBA32F
		}
		break;
	}

	width = srcWidth;
	height = srcHeight;
}

static inline void getBufferImage(const GLint target, const GLenum attachment, const char* filename)
	{
		GLint srcType      = 0;
		GLint srcTex       = 0;
		GLint srcLevel     = 0;
		GLint srcWidth     = 0;
		GLint srcHeight    = 0;
		GLint srcFormat    = 0;
		GLint srcCubeFace  =  0 ;

		glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, attachment, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &srcType);					//GL_TEXTURE = 5890 = 0x1702
		glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, attachment, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &srcTex);
		glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, attachment, GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL, &srcLevel);
		glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, attachment, GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE, &srcCubeFace);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(target, srcTex);

		if (srcCubeFace == 0)
		{
			glGetTexLevelParameteriv(target, srcLevel, GL_TEXTURE_WIDTH, &srcWidth);
			glGetTexLevelParameteriv(target, srcLevel, GL_TEXTURE_HEIGHT, &srcHeight);
			glGetTexLevelParameteriv(target, srcLevel, GL_TEXTURE_INTERNAL_FORMAT, &srcFormat); // get internal format type of GL texture, e.g. GL_RGBA32F
		}
		else
		{
			glGetTexLevelParameteriv(srcCubeFace, srcLevel, GL_TEXTURE_WIDTH, &srcWidth);
			glGetTexLevelParameteriv(srcCubeFace, srcLevel, GL_TEXTURE_HEIGHT, &srcHeight);
			glGetTexLevelParameteriv(srcCubeFace, srcLevel, GL_TEXTURE_INTERNAL_FORMAT, &srcFormat); // get internal format type of GL texture, e.g. GL_RGBA32F
		}

		int srcSize = srcWidth * srcHeight;
		unsigned char* dstBuffer = new unsigned char[4 * srcWidth * srcHeight];

		float minF, maxF;

		switch (srcFormat)
		{
			case GL_RGBA8:
			{
				srcSize *= 4;
				if (srcCubeFace == 0)
					glGetTexImage(target, 0, GL_BGRA, GL_UNSIGNED_BYTE, dstBuffer);
				else
					glGetTexImage(srcCubeFace, 0, GL_BGRA, GL_UNSIGNED_BYTE, dstBuffer);
			}
			break;
			case GL_RGBA32F:
			{
				srcSize *= 4;
				float *srcBuffer = new float[srcSize];
				int srcIdx, dstIdx;
				if (srcCubeFace == 0)
					glGetTexImage(target, 0, GL_RGBA, GL_FLOAT, srcBuffer);
				else
					glGetTexImage(srcCubeFace, 0, GL_RGBA, GL_FLOAT, srcBuffer);

				for (int i = 0; i < srcHeight; i++)
				{
					for (int j = 0; j < srcWidth * 4; j++)
					{
						srcIdx = (i * srcWidth * 4) + j;
						dstIdx = (i * srcWidth * 4) + j;
						dstBuffer[dstIdx] = (unsigned char)(255 * srcBuffer[srcIdx]);
					}
				}
				delete[] srcBuffer;
			}
			break;
			case GL_DEPTH_COMPONENT32F:
			{
				float* srcBuffer = new float[srcSize];
				int srcIdx, dstIdx;
				if (srcCubeFace == 0)
					glGetTexImage(target, 0, GL_DEPTH_COMPONENT, GL_FLOAT, srcBuffer);
				else
					glGetTexImage(srcCubeFace, 0, GL_DEPTH_COMPONENT, GL_FLOAT, srcBuffer);

				for (int i = 0; i < srcHeight; i++)
				{
					for (int j = 0; j < srcWidth; j++)
					{
						int srcIdx = (i * srcWidth) + j;
						if ((j == 0) && (i == 0))
						{
							minF = maxF = srcBuffer[srcIdx];
						}
						else
						{
							minF = MINIMUM(srcBuffer[srcIdx], minF);
							maxF = MAXIMUM(srcBuffer[srcIdx], maxF);
						}
					}
				}

				for (int i = 0; i < srcHeight; i++)
				{
					for (int j = 0; j < srcWidth; j++)
					{
						srcIdx = (i * srcWidth) + j;
						dstIdx = 4 * ((i*srcWidth) + j);

						// This is just to make a more visible image. Scale so that the range (minVal, 1.0) maps to (0.0, 1.0). This probably should be tweaked for different light configurations.
						float minVal = minF - 0.0001f;
						float scale = (srcBuffer[srcIdx] - minVal) / (1.0f - minVal);
						unsigned char val = (unsigned char)(scale * 255);
						dstBuffer[dstIdx] = val;
						dstBuffer[dstIdx + 1] = val;
						dstBuffer[dstIdx + 2] = val;
						dstBuffer[dstIdx + 3] = 0xff;
					}
				}
				delete[] srcBuffer;
			}
			break;
		}


		FIBITMAP* image = FreeImage_ConvertFromRawBits(dstBuffer, srcWidth, srcHeight, 4 * srcWidth, 32, 0xFF0000, 0x00FF00, 0x0000FF, false);
		FIBITMAP *image2 = FreeImage_ConvertTo24Bits(image);
		FreeImage_Save(FIF_BMP, image2, filename);
		FreeImage_Unload(image);
		FreeImage_Unload(image2);

		delete[] dstBuffer;
	}

//
//static inline bool saveBufferToFile(const DynArray<Buffer>* buffers, const char *fileName)
//{
//	FILE *f;
//	errno_t error;
//
//	if (error = fopen_s(&f, fileName, "wb") != 0) return false;
//
//	for (unsigned int i = 0; i<buffers->size(); i++)
//	{
//		if (buffers->at(i).m_data == 0) continue;
//		fwrite(&(buffers->at(i).m_sizeInBytes), sizeof(unsigned int), 1, f);
//		fwrite(&(buffers->at(i).m_noElements), sizeof(unsigned int), 1, f);
//		fwrite(&(buffers->at(i).m_elementSizeInBytes), sizeof(unsigned int), 1, f);
//		fwrite(buffers->at(i).m_data, buffers->at(i).m_sizeInBytes, 1, f);
//	}
//
//	fflush(f);
//	fclose(f);
//	return true;
//}
//
//static inline DynArray<Buffer>* loadBuffersFromFile(const char *fileName)
//{
//	FILE *f;
//	errno_t error;
//
//	if (error = fopen_s(&f, fileName, "rb") != 0) return nullptr;
//
//	_fseeki64(f, 0L, SEEK_END);
//	long long int totalSize = _ftelli64(f);
//	_fseeki64(f, 0L, SEEK_SET);
//
//	DynArray<Buffer> *buffers = new DynArray<Buffer>(4);
//	long long total = 0;
//	while (total < totalSize)
//	{
//		Buffer b;
//		total += fread(&b.m_sizeInBytes, sizeof(unsigned int), 1, f);
//		total += fread(&b.m_noElements, sizeof(unsigned int), 1, f);
//		total += fread(&b.m_elementSizeInBytes, sizeof(unsigned int), 1, f);
//
//		b.m_data = malloc(b.m_sizeInBytes);
//		total += fread(b.m_data, b.m_sizeInBytes, 1, f);
//
//		buffers->push_back(b);
//	}
//	fclose(f);
//	return buffers;
//}

#endif
