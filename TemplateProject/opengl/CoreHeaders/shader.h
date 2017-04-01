////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	shader.h
//
// summary:	Declaration and implementation of Shader class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __SHADER_H_
#define __SHADER_H_

#include <glew.h>
#include <fstream>
#include <string>
#include <regex>

#include "../code/util.h"

using namespace std;

enum ShaderState
{
	SS_UNKNOWN = 0, 
	SS_LOADED,
	SS_CREATED,
	SS_COMPILED
};

class __declspec( dllexport ) Shader
{
friend class ShaderProgram;

private:
	static const unsigned int SHADER_MAX_LINES = 512;
	static const unsigned int SHADER_MAX_LINE_LENGTH = 256;			//= 255 + NULL terminate


protected:
	ShaderState m_shaderState;
	GLenum m_shaderType;
	GLuint m_shaderObject;

public:
	Shader(GLenum  shaderType) : m_shaderType(shaderType), m_shaderState(SS_UNKNOWN), m_shaderObject(0) {}
	virtual ~Shader() { if (m_shaderObject >0) glDeleteShader(m_shaderObject); }
	   
	inline GLhandleARB getShader()  { return m_shaderObject; }
		 
	void create();
	void openFromFile(const char *filename);
	void openFromArray(const char *data);
	char *getInfoLog();

	friend class ShaderProgram;
};


inline void Shader::create()
{
	if (m_shaderState == SS_CREATED) return;
	m_shaderObject = glCreateShader(m_shaderType);   //glCreateShaderObjectARB(shaderType);
	m_shaderState = SS_CREATED;
}

inline char *Shader::getInfoLog()
{
	char *infoLog;
	int infoLogLength;
	glGetShaderiv(m_shaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);
	infoLog = new char[infoLogLength + 1];
	glGetShaderInfoLog(m_shaderObject, infoLogLength, &infoLogLength, infoLog);
	return infoLog;
}

inline void Shader::openFromFile(const char *filename)
{
	ifstream input(filename);

	if (!input.is_open())
		throw "Shader::exception(47):  unable to load shader file!";

	if (m_shaderObject == 0)
		create();

	std::string programStr;
	std::string line;
	std::regex includeRegex("#include\\s*\"(.*)\"");
	std::smatch regexMatch;
	std::string shaderPath(filename);
	shaderPath = shaderPath.substr(0, shaderPath.find_last_of('/'));

	while (std::getline(input, line))
	{
		if (std::regex_match(line, regexMatch, includeRegex))
		{
			std::string file = regexMatch[1].str();
			programStr += loadFile(shaderPath + "/" + file);
		}
		else
		{
			programStr += line;
		}

		programStr += '\n';
	}

	input.close();

	const GLchar* stringData = programStr.c_str();
	glShaderSource(m_shaderObject, 1, &stringData, NULL);
	m_shaderState = SS_LOADED;

	glCompileShader(m_shaderObject);
	int status = 0;
	glGetShaderiv(m_shaderObject, GL_COMPILE_STATUS, (int *)&status);
	if (!status)
	{
		char errorMsg[1024] = { '\0' };
		glGetShaderInfoLog(m_shaderObject, 1024, 0, errorMsg);
		printf("ERROR (%s): %s\n", filename, errorMsg);
		throw "Shader::exception(80):  shader did not compile!";
	}

	m_shaderState = SS_COMPILED;
}

inline void Shader::openFromArray(const char *data)
{
	if (data)
		create();

	const char**programPtr = (const char **)&data;
	glShaderSource(m_shaderObject, 1, programPtr, NULL);
	m_shaderState = SS_LOADED;

	glCompileShaderARB(m_shaderObject);

	int status = 0;
	glGetShaderiv(m_shaderObject, GL_COMPILE_STATUS, (int *)&status);
	if (!status)
		throw "Shader::exception(80):  shader did not compile!";

	m_shaderState = SS_COMPILED;
}

#endif

