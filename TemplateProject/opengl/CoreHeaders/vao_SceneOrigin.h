////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	vao_SceneOrigin.h
//
// summary:	Declaration and implementation of Vertex Array Object class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __VAO_SCENEORIGIN_H_
#define __VAO_SCENEORIGIN_H_

#include "vao.h"

class VAO_SceneOrigin : public VAO
{
public:
	VAO_SceneOrigin() : VAO() {}
	virtual ~VAO_SceneOrigin(void) {}

	virtual void init()
	{
		float tmpVertices[6 * 7] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,	//blue X-axis
									1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
									0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,	//green Y-axis
									0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
									0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,	//red Z-axis
									0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f };

		GLuint tmpBuffer;
		glGenBuffers(1, &tmpBuffer);
		initBuffer(tmpBuffer, tmpVertices, 6 * 7 * sizeof(float), GL_ARRAY_BUFFER, GL_STATIC_DRAW);
		m_buffers.push_back(tmpBuffer);

		glGenVertexArrays(1, &m_object);
		glBindVertexArray(m_object);
		glBindBuffer(GL_ARRAY_BUFFER, tmpBuffer);
		glEnableVertexAttribArray(0); // position
		glEnableVertexAttribArray(1); // color
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), 0);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (GLvoid*)(3 * sizeof(float)));

		//Unbind
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		m_isInitialized = true;
	}
};

#endif