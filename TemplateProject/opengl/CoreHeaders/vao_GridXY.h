////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	vao_GridXY.h
//
// summary:	Declaration and implementation of Vertex Array Object class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __VAO_GRIDXY_H_
#define __VAO_GRIDXY_H_

#include "vao.h"

class VAO_GridXY : public VAO
{
public:
	const int m_noPositiveCells;

public:
	VAO_GridXY() : VAO(), m_noPositiveCells(10) {}
	virtual ~VAO_GridXY(void) {}

	virtual void init()
	{
		int noLines = 2 * m_noPositiveCells + 1;									//no lines in a single direction

		float *tmpData = (float*)malloc(4 * noLines * 7 * sizeof(float));			//xyzwrgba for every vertex
		float *tmpPtr = tmpData;
		for (int i = 0; i<noLines; i++)
		{
			//vertical line
			*tmpPtr++ = (float)-m_noPositiveCells + i;		*tmpPtr++ = (float)-m_noPositiveCells;		*tmpPtr++ = 0.0f;		//xyz	
			*tmpPtr++ = .2f;		*tmpPtr++ = 0.2f;		*tmpPtr++ = 0.2f;	*tmpPtr++ = .3f;								//rgba

			*tmpPtr++ = (float)-m_noPositiveCells + i;		*tmpPtr++ = (float)m_noPositiveCells;		*tmpPtr++ = 0.0f;		//xyw
			*tmpPtr++ = .2f;		*tmpPtr++ = 0.2f;		*tmpPtr++ = 0.2f;	*tmpPtr++ = .3f;								//rgba

			//horizontal line
			*tmpPtr++ = (float)-m_noPositiveCells;		*tmpPtr++ = (float)-m_noPositiveCells + i;		*tmpPtr++ = 0.0f;			//xyw
			*tmpPtr++ = .2f;		*tmpPtr++ = 0.2f;		*tmpPtr++ = 0.2f;	*tmpPtr++ = .3f;								//rgba

			*tmpPtr++ = (float)m_noPositiveCells;		*tmpPtr++ = (float)-m_noPositiveCells + i;		*tmpPtr++ = 0.0f;			//xyz
			*tmpPtr++ = .2f;		*tmpPtr++ = 0.2f;		*tmpPtr++ = 0.2f;	*tmpPtr++ = .3f;								//rgba
		}

		GLuint tmpBuffer;
		glGenBuffers(1, &tmpBuffer);
		initBuffer(tmpBuffer, tmpData, 4 * noLines * 7 * sizeof(float), GL_ARRAY_BUFFER, GL_STATIC_DRAW);
		m_buffers.push_back(tmpBuffer);

		//printBuffer<float>(tmpData, 4*(2*noPositiveCells+1)*7, 7, "tmpData");

		free(tmpData);

		glGenVertexArrays(1, &m_object);
		glBindVertexArray(m_object);
		glBindBuffer(GL_ARRAY_BUFFER, m_buffers[0]);
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