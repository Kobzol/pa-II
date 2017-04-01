////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	bufferObject.h
//
// summary:	Declaration and implementation of Buffer Object class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __BUFEFEROBJECT_H_
#define __BUFEFEROBJECT_H_

#include "stdafx.h"
#include <string.h>
#include <dynArray.h>
#include "sceneSetting.h"
#include "helper.h"

#define DEFAULT_BUFFER_NAME_LENGTH 64

class BufferObject
{
public:
	GLuint m_object;						
	DynArray<GLuint> m_buffers;
	bool m_isInitialized;

public:
	BufferObject() : m_object(0), m_buffers(DynArray<GLuint>(2)), m_isInitialized(false) { }
	virtual ~BufferObject(void) {}

	virtual void init()	{ m_isInitialized = false; }
	
	virtual void bind(){}
	virtual void unbind(){}
	virtual void clear();
};

inline void BufferObject::clear()
{
	if (m_isInitialized)
	{
		glDeleteBuffers(m_buffers.size(), m_buffers.front());
		m_buffers.clear();
	}
}

#endif