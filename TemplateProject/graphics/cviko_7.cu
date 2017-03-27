#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <glew.h>
#include <freeglut.h>
#include <cudaDefs.h>
#include <imageManager.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include "imageKernels.cuh"
#include "../cudautil.cuh"
#include "../opengl/CoreHeaders/shader.h"
#include "../opengl/CoreHeaders/shaderProgram.h"
#include "../opengl/code/uniform.h"

#define BLOCK_DIM 8

//CUDA variables
static unsigned int imageWidth;
static unsigned int imageHeight;
static unsigned int imageBPP;		//Bits Per Pixel = 8, 16, 24, or 32 bit
static unsigned int imagePitch;

static cudaGraphicsResource_t cudaPBOResource;
static cudaGraphicsResource_t cudaTexResource;
texture<uchar4, 2, cudaReadModeElementType> cudaTexRef;
static cudaChannelFormatDesc cudaTexChannelDesc;
static KernelSetting ks;
static unsigned char someValue = 0;

//OpenGL
static unsigned int pboID;
static unsigned int textureID;
static unsigned int vaoID;
static unsigned int vboID;
static ShaderProgram* program;

static unsigned int viewportWidth = 1024;
static unsigned int viewportHeight = 1024;

#pragma region CUDA Routines

__global__ void applyFilter(const unsigned char someValue, const unsigned int pboWidth, const unsigned int pboHeight, unsigned char *pbo)
{
	//TODO 9: Create a filter that replaces Red spectrum of RGBA pbo such that RED=someValue
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	uchar4* ptr = (uchar4*) pbo;
	uchar4 value = tex2D(cudaTexRef, x, y);

	ptr[y * pboWidth + x] = make_uchar4(someValue, value.y, value.z, value.w);
}

void cudaWorker()
{
	cudaArray* array;
	//TODO 3: Map cudaTexResource
	cudaGraphicsMapResources(1, &cudaTexResource, 0);
	
	//TODO 4: Get Mapped Array of cudaTexResource
	cudaGraphicsSubResourceGetMappedArray(&array, cudaTexResource, 0, 0);
	
	//TODO 5: Get cudaTexChannelDesc from previously obtained array
	cudaGetChannelDesc(&cudaTexChannelDesc, array);

	//TODO 6: Binf cudaTexRef to array
	cudaBindTextureToArray(&cudaTexRef, array, &cudaTexChannelDesc);
	checkError();

	unsigned char *pboData;
	size_t pboSize;
	//TODO 7: Map cudaPBOResource
	cudaGraphicsMapResources(1, &cudaPBOResource, 0);
	checkError();
	
	//TODO 7: Map Mapped pointer to cudaPBOResource data
	cudaGraphicsResourceGetMappedPointer((void**) &pboData, &pboSize, cudaPBOResource);
	checkError();
		
	//TODO 8: Set KernelSetting variable ks (dimBlock, dimGrid, etc.) such that block will have BLOCK_DIM x BLOCK_DIM threads
	ks.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	ks.dimGrid = dim3(getNumberOfParts(imageWidth, BLOCK_DIM), getNumberOfParts(imageHeight, BLOCK_DIM), 1);

	//Calling applyFileter kernel
	someValue++;
	if (someValue>255) someValue = 0;
	applyFilter<<<ks.dimGrid, ks.dimBlock>>>(someValue, imageWidth, imageHeight, pboData);

	//Following code release mapped resources, unbinds texture and ensures that PBO data will be coppied into OpenGL texture. Do not modify following code!
	cudaUnbindTexture(&cudaTexRef);
	cudaGraphicsUnmapResources(1, &cudaPBOResource, 0);
	cudaGraphicsUnmapResources(1, &cudaTexResource, 0);
	
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pboID);
	glBindTexture( GL_TEXTURE_2D, textureID);
	glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);   //Source parameter is NULL, Data is coming from a PBO, not host memory
}

void initCUDAtex()
{
	cudaGLSetGLDevice(0);
	checkError();

	//CUDA Texture settings
	cudaTexRef.normalized = false;						//Otherwise TRUE to access with normalized texture coordinates
	cudaTexRef.filterMode = cudaFilterModePoint;		//Otherwise texRef.filterMode = cudaFilterModeLinear; for Linear interpolation of texels
	cudaTexRef.addressMode[0] = cudaAddressModeClamp;	//No repeat texture pattern
	cudaTexRef.addressMode[1] = cudaAddressModeClamp;	//No repeat texture pattern

	//TODO 1: Register OpenGL texture to CUDA resource
	cudaGraphicsGLRegisterImage(&cudaTexResource, textureID, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);
	checkError();

	//TODO 2: Register PBO to CUDA resource
	cudaGraphicsGLRegisterBuffer(&cudaPBOResource, pboID, cudaGraphicsRegisterFlagsWriteDiscard);
	checkError();

	glGenVertexArrays(1, &vaoID);
	glBindVertexArray(vaoID);
	glGenBuffers(1, &vboID);

	glBindBuffer(GL_ARRAY_BUFFER, vboID);
	glEnableVertexAttribArray(0); // position
	glEnableVertexAttribArray(1); // tex coords.
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)(3 * sizeof(float)));

	GLfloat vertices[] = 
	{
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,	// left bottom
		1.0f, -1.0f, 0.0f, 1.0f, 0.0f,	// right bottom
		1.0f,  1.0f, 0.0f, 1.0f, 1.0f,	// right up
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,	// left bottom
		1.0f,  1.0f, 0.0f, 1.0f, 1.0f,	// right up
		-1.0f, 1.0f, 0.0f, 0.0f, 1.0f	// left up
	};

	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	//Unbind
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//std::string vertexCode = loadFile("opengl/Resources/shaders/screen_v3_t2.vert");
	//std::string fragmentCode = loadFile("opengl/Resources/shaders/screen_v3_t2.frag");

	Shader vertexShader(GL_VERTEX_SHADER);
	vertexShader.openFromFile("opengl/Resources/shaders/screen_v3_t2.vert");
	
	Shader fragmentShader(GL_FRAGMENT_SHADER);
	fragmentShader.openFromFile("opengl/Resources/shaders/screen_v3_t2.frag");

	program = new ShaderProgram(&vertexShader, &fragmentShader);
	program->enable();
	Uniform<int>::bind("texSampler", program->m_programObject, 0);
}

void releaseCUDA()
{
	cudaGraphicsUnregisterResource(cudaPBOResource);
	cudaGraphicsUnregisterResource(cudaTexResource);
}
#pragma endregion

#pragma region OpenGL Routines - DO NOT MODIFY THIS SECTION !!!

void loadTexture(const char* imageFileName)
{
	FreeImage_Initialise();
	FIBITMAP *tmp = ImageManager::GenericLoader(imageFileName, 0);

	imageWidth = FreeImage_GetWidth(tmp);
	imageHeight = FreeImage_GetHeight(tmp);
	imageBPP = FreeImage_GetBPP(tmp);
	imagePitch = FreeImage_GetPitch(tmp);

	//OpenGL Texture
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1,&textureID);
	glBindTexture( GL_TEXTURE_2D, textureID);

	//WARNING: Just some of inner format are supported by CUDA!!!
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, imageWidth, imageHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, FreeImage_GetBits(tmp));
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	FreeImage_Unload(tmp);
}

void preparePBO()
{
	glGenBuffers(1, &pboID);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);												// Make this the current UNPACK buffer (OpenGL is state-based)
	glBufferData(GL_PIXEL_UNPACK_BUFFER, imageWidth * imageHeight * 4, NULL,GL_DYNAMIC_COPY);	// Allocate data for the buffer. 4-channel 8-bit image
}

void my_display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, textureID);

	//I know this is a very old OpenGL, but we want to practice CUDA :-)
        //Now it will be a wasted time to learn you current features of OpenGL. Sorry for that however, you can visit my second seminar dealing with Computer Graphics (CG2).
	/*glBegin(GL_QUADS);

	glTexCoord2d(0,0);		glVertex2d(0,0);
	glTexCoord2d(1,0);		glVertex2d(viewportWidth, 0);
	glTexCoord2d(1,1);		glVertex2d(viewportWidth, viewportHeight);
	glTexCoord2d(0,1);		glVertex2d(0, viewportHeight);

	glEnd();*/

	glBindVertexArray(vaoID);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);

	glDisable(GL_TEXTURE_2D);

	glFlush();			
	glutSwapBuffers();
}

void my_resize(GLsizei w, GLsizei h)
{
	viewportWidth=w; 
	viewportHeight=h; 

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0,0,viewportWidth,viewportHeight);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0,viewportWidth, 0,viewportHeight);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glutPostRedisplay();
}

void my_idle() 
{
	cudaWorker();
	glutPostRedisplay();
}


void initGL(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(viewportWidth,viewportHeight);
	glutInitWindowPosition(0,0);
	glutCreateWindow(":-)");

	glutDisplayFunc(my_display);
	glutReshapeFunc(my_resize);
	glutIdleFunc(my_idle);
	glutSetCursor(GLUT_CURSOR_CROSSHAIR);

	// initialize necessary OpenGL extensions
	glewInit();

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glShadeModel(GL_SMOOTH);
	glDisable(GL_CULL_FACE);
	glViewport(0,0,viewportWidth,viewportHeight);

	glFlush();
}

void releaseOpenGL()
{
	if (textureID > 0)
		glDeleteTextures(1, &textureID);
	if (pboID > 0)
		glDeleteBuffers(1, &pboID);
}

#pragma endregion

void releaseResources()
{
	releaseCUDA();
	releaseOpenGL();
}

void cviko7(int argc, char** argv)
{
	initGL(argc, argv);

	loadTexture("graphics/lena.png");
	preparePBO();
	initCUDAtex();

	//start rendering mainloop
    glutMainLoop();
    atexit(releaseResources);
}
