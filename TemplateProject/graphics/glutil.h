#pragma once

#include <glew.h>

#include <string>
#include <iostream>

#define GL_CHECK_ERRORS() OpenGLHelper::glCheckError(__FILE__, __LINE__);
#define GL_CHECK_COMPILATION(shader) OpenGLHelper::glCheckCompilation(shader, __FILE__, __LINE__);
#define GL_CHECK_LINKAGE(program) OpenGLHelper::glCheckLinkage(program, __FILE__, __LINE__);
#define GL_PRINT_ERROR(message) OpenGLHelper::glPrintError(message, __FILE__, __LINE__);

class OpenGLHelper
{
public:
	inline static void glPrintError(std::string message, const char* filename, int line)
	{
		std::cerr << "OpenGL Error: " << message << " in " << filename << ":" << line << std::endl;
	}
	inline static void glClearError()
	{
		glGetError();
	}
	static bool glCheckError(const char* filename, int line)
	{
#ifdef _DEBUG
		GLenum error = glGetError();

		if (error != GL_NO_ERROR)
		{
			std::string message;
			switch (error)
			{
			case GL_INVALID_ENUM:
				message = "GL_INVALID_ENUM";
				break;
			case GL_INVALID_VALUE:
				message = "GL_INVALID_VALUE";
				break;
			case GL_INVALID_OPERATION:
				message = "GL_INVALID_OPERATION";
				break;
			case GL_STACK_OVERFLOW:
				message = "GL_STACK_OVERFLOW";
				break;
			case GL_STACK_UNDERFLOW:
				message = "GL_STACK_UNDERFLOW";
				break;
			case GL_OUT_OF_MEMORY:
				message = "GL_OUT_OF_MEMORY";
				break;
			default:
				message = "UNKNOWN_ERROR";
				break;
			}

			OpenGLHelper::glPrintError(message, filename, line);

			return false;
		}
#endif
		return true;
	}
	static bool glCheckCompilation(GLuint shader, const char* filename, int line)
	{
		GLint status;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

		if (status != GL_TRUE)
		{
			char buffer[512];
			glGetShaderInfoLog(shader, sizeof(buffer), nullptr, buffer);

			OpenGLHelper::glPrintError(std::string(buffer), filename, line);

			return false;
		}
		else return true;
	}
	static bool glCheckLinkage(GLuint program, const char* filename, int line)
	{
		GLint status;
		glGetProgramiv(program, GL_LINK_STATUS, &status);

		if (status != GL_TRUE)
		{
			char buffer[512];
			glGetProgramInfoLog(program, sizeof(buffer), nullptr, buffer);

			OpenGLHelper::glPrintError(std::string(buffer), filename, line);

			return false;
		}
		else return true;
	}
private:
	OpenGLHelper()
	{

	}
};
