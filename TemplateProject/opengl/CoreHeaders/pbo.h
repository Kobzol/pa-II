////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	vao.h
//
// summary:	Declaration and implementation of Vertex Array Object class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __PBO_H_
#define __PBO_H_

#include "stdafx.h"
#include "bufferObject.h"

class PBO : public BufferObject
{
public:
	const GLenum m_target;
	const GLenum m_access;
	const unsigned int m_sizeInBytes;

public:
	PBO(const GLenum target, const GLenum access, const unsigned int sizeInBytes) : BufferObject(), m_target(target), m_access(access), m_sizeInBytes(sizeInBytes) {}

	virtual ~PBO(){ if (m_object) glDeleteBuffers(1, &m_object); }
	
	virtual void init();

	virtual void bind();
	virtual void unbind();
};

inline void PBO::init()
{
	glGenBuffers(1, &m_object);
	glBindBufferARB(m_target, m_object);
	initBuffer(m_object, nullptr, m_sizeInBytes, m_target, m_access);
	
}

inline void PBO::bind()
{
	if (m_isInitialized)
		glBindBuffer(m_target, m_object);
}

inline void PBO::unbind()
{
	glBindBuffer(m_target, 0);
}


#endif