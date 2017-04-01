////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	material.h
//
// summary:	Declaration and implementation of Material structure
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __MATERIAL_H_
#define __MATERIAL_H_

#include <string.h>
#include "shaderProgram.h"

struct Material
{
	float m_ambient[4];
	float m_diffuse[4];
	float m_specular[4];	
	float m_transparency;
	float m_shininess;
	int m_illumination;
	GLuint m_diffuseTextureGL;

	char m_name[256];
	char m_diffuseTexture[256];

	Material()
	{
		m_ambient[0] = m_ambient[1] = m_ambient[2] = 0.1f; m_ambient[3] = 1.0f;
		m_diffuse[0] = m_diffuse[1] = m_diffuse[2] = 0.0f; m_diffuse[3] = 1.0f;
		m_specular[0] = m_specular[1] = m_specular[2] = 0.2f; m_specular[3] = 1.0f;
		m_transparency = 0.0f;
		m_shininess = 10;
		m_illumination = 0;
		m_diffuseTextureGL = 0;
		memset(m_name, '\0', 256);
		memset(m_diffuseTexture, '\0', 256);
	}

	inline void setName(const char *_name)
	{
		strcpy_s(m_name, 256, _name);
	}

	inline void operator=(const Material &other)
	{
		memcpy_s(this, sizeof(Material), &other, sizeof(Material));
	}

	inline bool operator==(const Material& other) const
	{
		if (m_transparency != other.m_transparency) return false;
		if (m_shininess != other.m_shininess) return false;
		if (m_illumination != other.m_illumination) return false;

		for (int i=0; i<4; i++)
		{
			if (m_ambient[i]!=other.m_ambient[i]) return false;
			if (m_diffuse[i]!=other.m_diffuse[i]) return false;
			if (m_specular[i]!=other.m_specular[i]) return false;
		}

		if (strcmp(m_diffuseTexture, other.m_diffuseTexture)!=0) return false;
		if (strcmp(m_name, other.m_name)!=0) return false;

		return true;
	}

	static void setShaderUniform(const Material* material, const ShaderProgram *spPtr, const char* variableName)
	{
		char tmp[128];
		size_t length = strlen(variableName);
		memcpy_s(tmp, 128, variableName, length);
		tmp[length++] = '.';
		tmp[length] = '\0';

		int uniform;

		memcpy_s(&tmp[length], 128, "ambient\0", 8);
		if ((uniform = glGetUniformLocation(spPtr->m_programObject, tmp))>=0)
			glUniform4fv(uniform, 1, material->m_ambient);

		memcpy_s(&tmp[length], 128, "diffuse\0", 8);
		if ((uniform = glGetUniformLocation(spPtr->m_programObject, tmp))>=0)
			glUniform4fv(uniform, 1, material->m_diffuse);

		memcpy_s(&tmp[length], 128, "specular\0", 9);
		if ((uniform = glGetUniformLocation(spPtr->m_programObject, tmp))>=0)
			glUniform4fv(uniform, 1, material->m_specular);

		memcpy_s(&tmp[length], 128, "transparency\0", 13);
		if ((uniform = glGetUniformLocation(spPtr->m_programObject, tmp))>=0)
			glUniform1f(uniform, material->m_transparency);

		memcpy_s(&tmp[length], 128, "shininess\0", 10);
		if ((uniform = glGetUniformLocation(spPtr->m_programObject, tmp))>=0)
			glUniform1f(uniform, material->m_shininess);

		memcpy_s(&tmp[length], 128, "illumination\0", 13);
		if ((uniform = glGetUniformLocation(spPtr->m_programObject, tmp))>=0)
			glUniform1i(uniform, material->m_illumination);

		memcpy_s(&tmp[length], 128, "hasDiffuseTexture\0", 18);
		if ((uniform = glGetUniformLocation(spPtr->m_programObject, tmp))>=0)
			glUniform1i(uniform, (material->m_diffuseTextureGL > 0) ? 1 : 0);

		//memcpy_s(&tmp[length], 128, "diffuseTexture\0", 15);
		//if ((uniform = glGetUniformLocation(spPtr->programObject, tmp))>=0)
		//	glUniform1ui(uniform, diffuseTextureGL);
		
		if ((uniform = glGetUniformLocation(spPtr->m_programObject, "diffuseTexture"))>=0)
			glUniform1i(uniform, 0);			//Texture unit is set = sampler will access GL_TEXTURE0 = texture unit 0. DO NOT SET the diffuse texture diffuseTextureGL!!! 
	}


};

#endif