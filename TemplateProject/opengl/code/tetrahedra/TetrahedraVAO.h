#pragma once

#include "vao.h"

class TetrahedraVAO : public VAO
{
public:
	TetrahedraVAO() : VAO() {}
	virtual ~TetrahedraVAO(void) {}

	virtual void init()
	{
		float tmpVertices[4 * 7] = {
			0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,	// red
			1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,	// blue
			0.5f, 0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,	// yellow
			0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,  // green
		};

		GLuint tmpBuffer;
		glGenBuffers(1, &tmpBuffer);
		initBuffer(tmpBuffer, tmpVertices, 4 * 7 * sizeof(float), GL_ARRAY_BUFFER, GL_STATIC_DRAW);
		m_buffers.push_back(tmpBuffer);

		unsigned int tmpIndices[4 * 3] = {
			2, 0, 1,
			2, 1, 3,
			2, 3, 0,
			1, 0, 3
		};

		glGenBuffers(1, &tmpBuffer);
		initBuffer(tmpBuffer, tmpIndices, 4 * 3 * sizeof(unsigned int), GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
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
