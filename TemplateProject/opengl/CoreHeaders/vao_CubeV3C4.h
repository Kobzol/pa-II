////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	vao_CubeV3C4.h
//
// summary:	Declaration and implementation of Vertex Array Object class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __VAO_CUBEV3C4_H_
#define __VAO_CUBEV3C4_H_

#include "vao.h"

class VAO_CubeV3C4 : public VAO
{
public:
	VAO_CubeV3C4() : VAO() {}
	virtual ~VAO_CubeV3C4(void) {}

	virtual void init()	
	{ 
									//x	  y		z	  r		g	  b		a	
		float tmpVertices[8 * 7] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
									1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
									1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
									0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f,
									0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
									1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f,
									1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
									0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };

		GLuint tmpBuffer;
		glGenBuffers(1, &tmpBuffer);
		initBuffer(tmpBuffer, tmpVertices, 8 * 7 * sizeof(float), GL_ARRAY_BUFFER, GL_STATIC_DRAW);
		m_buffers.push_back(tmpBuffer);

		unsigned int tmpIndices[6 * 4] = { 0, 3, 2, 1,
			4, 5, 6, 7,
			2, 3, 7, 6,
			0, 1, 5, 4,
			1, 2, 6, 5,
			0, 4, 7, 3 };

		glGenBuffers(1, &tmpBuffer);
		initBuffer(tmpBuffer, tmpIndices, 6 * 4 * sizeof(unsigned int), GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
		m_buffers.push_back(tmpBuffer);

		glGenVertexArrays(1, &m_object);
		glBindVertexArray(m_object);
		glBindBuffer(GL_ARRAY_BUFFER, m_buffers[0]);
		glEnableVertexAttribArray(0); // position
		glEnableVertexAttribArray(1); // color
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), 0);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (GLvoid*)(3 * sizeof(float)));
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffers[1]);

		//Unbind
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		m_isInitialized = true; 
	}
};

#endif