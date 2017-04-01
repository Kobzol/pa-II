////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	fbo.h
//
// summary:	Declaration and implementation of Frame Buffer Object
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __FBO_H_
#define __FBO_H_

#include "stdafx.h"
#include "bufferObject.h"

class FBO : public BufferObject
{
protected:
	bool m_reinitOnResize;

public:
	unsigned int m_screen[2];

public:
	FBO() : BufferObject(), m_reinitOnResize(false) {}
	virtual ~FBO();

	virtual void bind();
	virtual void unbind();
	virtual void clear();
	virtual void init();
	virtual void resize();

	GLuint createImmutableStorageTexture(const GLint target, const GLint internalFormat, const int width, const int height) const;
	GLuint createMutableStorageTexture(const GLint target, const GLint internalFormat, const GLint format,  const int width, const int height) const;
};

inline FBO::~FBO()
{
	int x = 0;
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &x);

	if (x == m_object)
		unbind();
	clear();
}

inline void FBO::init()
{
	m_isInitialized = false;			//Inherited classes should implement own init() method and set m_isInitialized = true;
}

inline void FBO::resize()
{
	if ((m_isInitialized) && (m_reinitOnResize))
	{
		SceneSetting *ss = SceneSetting::GetInstance();

		bind();

		GLint srcWidth = 0;
		GLint srcHeight = 0;
		getBufferSize(GL_TEXTURE_2D, GL_COLOR_ATTACHMENT0, srcWidth, srcHeight);

		if ((ss->m_screen[0] != srcWidth) || (ss->m_screen[1] != srcHeight))
		{
			clear();
			m_isInitialized = false;
			init();
		}
		else
		{
			unbind();
		}
	}
}

inline void FBO::clear()
{
	if (m_isInitialized)
	{
		m_isInitialized = false;
		glDeleteFramebuffers(1, &m_object);
	}
}

inline void FBO::bind()
{
	if (m_isInitialized)
		glBindFramebuffer(GL_FRAMEBUFFER, m_object);
}

inline void FBO::unbind()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

inline GLuint FBO::createImmutableStorageTexture(const GLint target, const GLint internalFormat, const int width, const int height) const
{
	//target: GL_TEXTURE_2D or GL_TEXTURE_CUBE_MAP

	GLuint id;

	glGenTextures(1, &id);
	glBindTexture(target, id);

	glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glTexStorage2D(target, 1, internalFormat, width, height);			//This should work for GL_TEXTURE_CUBE_MAP but it does not.
		
	return id;
}

inline GLuint FBO::createMutableStorageTexture(const GLint target, const GLint internalFormat, const GLint format, const int width, const int height) const
{
	GLuint id;

	glGenTextures(1, &id);
	glBindTexture(target, id);

	glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	if (target == GL_TEXTURE_CUBE_MAP)
	{
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 0, 0, internalFormat, width, height, 0, format, GL_FLOAT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 1, 0, internalFormat, width, height, 0, format, GL_FLOAT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 2, 0, internalFormat, width, height, 0, format, GL_FLOAT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 3, 0, internalFormat, width, height, 0, format, GL_FLOAT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 4, 0, internalFormat, width, height, 0, format, GL_FLOAT, NULL);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 5, 0, internalFormat, width, height, 0, format, GL_FLOAT, NULL);
	}
	else
		glTexImage2D(target, 0, internalFormat, width, height, 0, format, GL_FLOAT, NULL);



		return id;
}

#endif