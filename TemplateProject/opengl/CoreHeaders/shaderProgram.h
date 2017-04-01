////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	shaderProgram.h
//
// summary:	Declaration and implementation of ShaderProgram class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __SHADERPROGRAM_H_
#define __SHADERPROGRAM_H_

#include "shader.h"

enum ShadersFlag : unsigned char
{
	SF_NONE = 0,
	SF_VERTEX = 1,
	SF_FRAGMENT = 2,
	SF_GEOMETRY = 4,
	SF_TESSELATION_CONTROL = 8,
	SF_TESSELATION_EVALUATION = 16,
	SF_ALL = SF_VERTEX | SF_FRAGMENT | SF_GEOMETRY | SF_TESSELATION_CONTROL | SF_TESSELATION_EVALUATION
};

class __declspec( dllexport ) ShaderProgram
{
public:
	GLuint m_programObject;

	Shader *m_vertexShader;
	Shader *m_fragmentShader;
	Shader *m_geometryShader;
	Shader *m_tesselationControlShader;
	Shader *m_tesselationEvaluationShader;
	ShadersFlag m_availableShaders;
	    
private:
	char *getInfoLog();
	    
private:
	inline void attachShader(Shader *s) { glAttachShader(m_programObject, s->m_shaderObject); }
	inline void detachShader(Shader *s) { glDetachShader(m_programObject, s->m_shaderObject); }
	inline void linkProgram();

	void printLogInfo();
	void validate();

public:
	ShaderProgram(Shader *_vertexShader, Shader *_fragmentShaderm);
	ShaderProgram(Shader *_vertexShader, Shader *_fragmentShaderm, Shader *_geometryShader);
	ShaderProgram::ShaderProgram(Shader *_vertexShader, Shader *_fragmentShader, Shader *_geometryShader, Shader *_tesselationControlShader, Shader *_tesselationEvaluationShader);
	~ShaderProgram();
	
	void attachShaders(ShadersFlag flag = SF_ALL);
	void detachShaders(ShadersFlag flag = SF_ALL);

	inline void enable() { disable(); glUseProgram(m_programObject); }
	inline void disable() { glUseProgram(0); }
};

inline ShaderProgram::ShaderProgram(Shader *_vertexShader, Shader *_fragmentShader)
{
	m_programObject = glCreateProgram();   //glCreateProgramObjectARB();
	m_availableShaders = SF_NONE;

	if (_vertexShader)
	{
		m_vertexShader = _vertexShader;
		m_availableShaders = (ShadersFlag)(m_availableShaders | SF_VERTEX);
	}
	if (_fragmentShader)
	{
		m_fragmentShader = _fragmentShader;
		m_availableShaders = (ShadersFlag)(m_availableShaders | SF_FRAGMENT);
	}

	attachShaders(m_availableShaders);
	linkProgram();
	detachShaders(m_availableShaders);
}

inline ShaderProgram::ShaderProgram(Shader *_vertexShader, Shader *_fragmentShader, Shader *_geometryShader)
{
	m_programObject = glCreateProgram();
	m_availableShaders = SF_NONE;

	if (_vertexShader)
	{
		m_vertexShader = _vertexShader;
		m_availableShaders = (ShadersFlag)(m_availableShaders | SF_VERTEX);
	}
	if (_fragmentShader)
	{
		m_fragmentShader = _fragmentShader;
		m_availableShaders = (ShadersFlag)(m_availableShaders | SF_FRAGMENT);
	}
	if (_geometryShader)
	{
		m_geometryShader = _geometryShader;
		m_availableShaders = (ShadersFlag)(m_availableShaders | SF_GEOMETRY);
	}

	attachShaders(m_availableShaders);
	linkProgram();
	detachShaders(m_availableShaders);
}

inline ShaderProgram::ShaderProgram(Shader *_vertexShader, Shader *_fragmentShader, Shader *_geometryShader, Shader *_tesselationControlShader, Shader *_tesselationEvaluationShader)
{
	m_programObject = glCreateProgram();
	m_availableShaders = SF_NONE;

	if (_vertexShader)
	{
		m_vertexShader = _vertexShader;
		m_availableShaders = (ShadersFlag)(m_availableShaders | SF_VERTEX);
	}
	if (_fragmentShader)
	{
		m_fragmentShader = _fragmentShader;
		m_availableShaders = (ShadersFlag)(m_availableShaders | SF_FRAGMENT);
	}
	if (_geometryShader)
	{
		m_geometryShader = _geometryShader;
		m_availableShaders = (ShadersFlag)(m_availableShaders | SF_GEOMETRY);
	}
	if (_tesselationControlShader)
	{
		m_tesselationControlShader = _tesselationControlShader;
		m_availableShaders = (ShadersFlag)(m_availableShaders | SF_TESSELATION_CONTROL);
	}
	if (_tesselationEvaluationShader)
	{
		m_tesselationEvaluationShader = _tesselationEvaluationShader;
		m_availableShaders = (ShadersFlag)(m_availableShaders | SF_TESSELATION_EVALUATION);
	}
	attachShaders(m_availableShaders);
	linkProgram();
	detachShaders(m_availableShaders);
}

inline ShaderProgram::~ShaderProgram()
{
	detachShaders(m_availableShaders);
	glDeleteProgram(m_programObject);
}

inline char *ShaderProgram::getInfoLog()
{
	char *infoLog;
	int infoLogLength;

	glGetProgramiv(m_programObject, GL_INFO_LOG_LENGTH, &infoLogLength);

	infoLog = new char[infoLogLength + 1];
	glGetProgramInfoLog(m_programObject, infoLogLength, &infoLogLength, infoLog);
	return infoLog;
}

inline void ShaderProgram::attachShaders(ShadersFlag flag)
{
	ShadersFlag mixFlag = (ShadersFlag)(flag & m_availableShaders);
	if (mixFlag & SF_VERTEX)	attachShader(m_vertexShader);
	if (mixFlag & SF_FRAGMENT)	attachShader(m_fragmentShader);
	if (mixFlag & SF_GEOMETRY)	attachShader(m_geometryShader);
	if (mixFlag & SF_TESSELATION_CONTROL)	attachShader(m_tesselationControlShader);
	if (mixFlag & SF_TESSELATION_EVALUATION)	attachShader(m_tesselationEvaluationShader);
}

inline void ShaderProgram::detachShaders(ShadersFlag flag)
{
	ShadersFlag mixFlag = (ShadersFlag)(flag & m_availableShaders);
	if (mixFlag & SF_VERTEX)	detachShader(m_vertexShader);
	if (mixFlag & SF_FRAGMENT)	detachShader(m_fragmentShader);
	if (mixFlag & SF_GEOMETRY)	detachShader(m_geometryShader);
	if (mixFlag & SF_TESSELATION_CONTROL)	detachShader(m_tesselationControlShader);
	if (mixFlag & SF_TESSELATION_EVALUATION)	detachShader(m_tesselationEvaluationShader);
}

inline void ShaderProgram::printLogInfo()
{
	int max_length = 2048;
	int actual_length = 0;
	char log[2048];
	glGetProgramInfoLog(m_programObject, max_length, &actual_length, log);
	printf("program info log for GL index %u:\n%s", m_programObject, log);
}

inline void ShaderProgram::validate()
{
	glValidateProgram(m_programObject);
	int params = -1;
	glGetProgramiv(m_programObject, GL_VALIDATE_STATUS, &params);
	if (GL_TRUE != params)
	{
		printf("program %i GL_VALIDATE_STATUS = GL_FALSE\n", m_programObject);
		printLogInfo();
		return;
	}
	printf("program %i GL_VALIDATE_STATUS = GL_TRUE\n", m_programObject);
}

inline void ShaderProgram::linkProgram()
{ 
	glLinkProgram(m_programObject); 
	
	int params = -1;
	glGetProgramiv(m_programObject, GL_LINK_STATUS, &params);
	if (GL_TRUE != params)
	{
		printf("program %i GL_VALIDATE_STATUS = GL_FALSE\n", m_programObject);
		printLogInfo();
		return;
	}
	validate(); 
}




#endif