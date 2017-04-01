#pragma once

#include "vao.h"

class QuadVAO : public VAO
{
public:
	QuadVAO() : VAO() {}
	virtual ~QuadVAO(void) {}

	virtual void init()
	{
		float tmpVertices[4 * 5] = {
			-0.5f,	-0.5f,	0.0f,	0.0f,	0.0f,
			0.5f,	-0.5f,	0.0f,	1.0f,	0.0f,
			0.5f,	0.5f,	0.0f,	1.0f,	1.0f,
			-0.5f,	0.5f,	0.0f,	0.0f,	1.0f,
		};

		GLuint tmpBuffer;
		glGenBuffers(1, &tmpBuffer);
		initBuffer(tmpBuffer, tmpVertices, sizeof(tmpVertices), GL_ARRAY_BUFFER, GL_STATIC_DRAW);
		m_buffers.push_back(tmpBuffer);

		glGenVertexArrays(1, &m_object);
		glBindVertexArray(m_object);
		glBindBuffer(GL_ARRAY_BUFFER, m_buffers[0]);
		glEnableVertexAttribArray(0); // position
		glEnableVertexAttribArray(1); // texture coordinate
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)(3 * sizeof(float)));

		//Unbind
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		m_isInitialized = true;
	}
};
