////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	ubo.h
//
// summary:	Declaration and implementation of Uniform Buffer Object structure
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __UBO_H_
#define __UBO_H_

#include <string.h>

#include "bufferObject.h"

/// <summary>	An UBO. 
/// 			In case of UBO class, the whole GL buffer objects is accessed via m_object variable of the BufferObject class. UBO does not contain any other sub-buffers, compared to e.g. FBO or VAO.</summary>
/// <remarks>	Gajdi, 7. 4. 2015. </remarks
class UBO : public BufferObject
{
public:
	GLuint m_bindingPoint;

public:
	UBO(GLuint bindingPoint) : BufferObject(), m_bindingPoint(bindingPoint) {}
	virtual ~UBO(){ if (m_object) glDeleteBuffers(1, &m_object); }

	virtual void bind();
	virtual void unbind();
	virtual void clear();
	
	template <typename T>
	void init(GLuint program, const std::string& uboName);

	template <typename T>
	void copy(const T* uboObject);

	virtual void getUniformInfo(){}
};

template <typename T>
inline void UBO::init(GLuint program, const std::string& uboName)
{
	if (m_isInitialized)
	{
		clear();
	}

	glGenBuffers(1, &m_object);
	m_isInitialized = true;

	this->bind();

	unsigned int block_index = glGetUniformBlockIndex(program, uboName.c_str());

	glBindBufferBase(GL_UNIFORM_BUFFER, this->m_bindingPoint, m_object);	// bind buffer to binding point
	glUniformBlockBinding(program, block_index, this->m_bindingPoint);	// connect buffer with uniform
	glBufferData(GL_UNIFORM_BUFFER, sizeof(T), NULL, GL_DYNAMIC_DRAW);

	this->unbind();
}

inline void UBO::clear()
{
	if (m_isInitialized)
	{
		m_isInitialized = false;
		glDeleteBuffers(1, &m_object);
	}
}

inline void UBO::bind()
{
	if (m_isInitialized)
		glBindBuffer(GL_UNIFORM_BUFFER, m_object);
}

inline void UBO::unbind()
{
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

template <typename T>
void UBO::copy(const T* uboObject)
{
	this->bind();

	GLvoid* mem = glMapBuffer(GL_UNIFORM_BUFFER, GL_WRITE_ONLY);
	memcpy(mem, uboObject, sizeof(T));
	glUnmapBuffer(GL_UNIFORM_BUFFER);

	this->unbind();
}

#endif
