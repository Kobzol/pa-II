////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	vao.h
//
// summary:	Declaration and implementation of Vertex Array Object class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __VAO_H_
#define __VAO_H_

#include "stdafx.h"
#include "bufferObject.h"
#include "eai.h"
#include "model.h"

class VAO : public BufferObject
{
public:
	DynArray<ELEMENT_ARRAY_INFO> *m_eai;			//Says how VAO will be drown

public:
	VAO() : BufferObject(), m_eai(nullptr) {}
	virtual ~VAO(void)
	{
		if (m_object>0)
		{
			glDeleteVertexArrays(1, &m_object);
			m_object = 0;
		}
		for (unsigned int i = 0; i<m_buffers.size(); i++)
		{
			glDeleteBuffers(1, &m_buffers[i]);
		}
		m_buffers.clear();

		if (m_eai)
			SAFE_DELETE(m_eai);
	}

	virtual void init()	{ m_isInitialized = false;}

	void createFromModel(Model *m)
	{
		DynArray<Buffer<PNT>> *geometry;
		m->createVAOGeometry(geometry, m_eai, false);

		GLuint tmpBuffer;
		Buffer<PNT> &gb = geometry->at(0);

		glGenVertexArrays(1, &m_object); 
		glBindVertexArray(m_object);
		glGenBuffers(1, &tmpBuffer); 

			glBindBuffer(GL_ARRAY_BUFFER, tmpBuffer);
			glEnableVertexAttribArray(0); // position
			glEnableVertexAttribArray(1); // normal
			glEnableVertexAttribArray(2); // tex coords.
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, gb.m_elementSizeInBytes, 0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, gb.m_elementSizeInBytes, (GLvoid*)(3*sizeof(float)));
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, gb.m_elementSizeInBytes, (GLvoid*)(6*sizeof(float)));

			initBuffer(tmpBuffer, gb.m_data, gb.m_sizeInBytes , GL_ARRAY_BUFFER, GL_STATIC_DRAW);
			m_buffers.push_back(tmpBuffer);	
			
		//Unbind
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0); 
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); 

		SAFE_DELETE(geometry); 

		m_isInitialized = true;
	}

	void createFromModelInterpolated(Model *m, Model* m2)
	{
		DynArray<Buffer<PNTP>> *geometry;
		m->createVAOGeometryInterpolated(geometry, m_eai, false, m2);

		GLuint tmpBuffer;
		Buffer<PNTP> &gb = geometry->at(0);

		glGenVertexArrays(1, &m_object);
		glBindVertexArray(m_object);
		glGenBuffers(1, &tmpBuffer);

		glBindBuffer(GL_ARRAY_BUFFER, tmpBuffer);
		glEnableVertexAttribArray(0); // position
		glEnableVertexAttribArray(1); // normal
		glEnableVertexAttribArray(2); // tex coords.
		glEnableVertexAttribArray(3); // position 2
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, gb.m_elementSizeInBytes, 0);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, gb.m_elementSizeInBytes, (GLvoid*)(3 * sizeof(float)));
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, gb.m_elementSizeInBytes, (GLvoid*)(6 * sizeof(float)));
		glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, gb.m_elementSizeInBytes, (GLvoid*)(9 * sizeof(float)));

		initBuffer(tmpBuffer, gb.m_data, gb.m_sizeInBytes, GL_ARRAY_BUFFER, GL_STATIC_DRAW);
		m_buffers.push_back(tmpBuffer);

		//Unbind
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		SAFE_DELETE(geometry);

		m_isInitialized = true;
	}


	void createFromModelWithTBN(Model *m)
	{
		DynArray<Buffer<PNT_TBN>> *geometry;
		m->createVAOGeometryWithTBN(geometry, m_eai, false);
		m->updateTBN(geometry, m_eai, false);

		GLuint tmpBuffer;
		Buffer<PNT_TBN> &gb = geometry->at(0);

		glGenVertexArrays(1, &m_object);
		glBindVertexArray(m_object);
		glGenBuffers(1, &tmpBuffer);

			glBindBuffer(GL_ARRAY_BUFFER, tmpBuffer);
			glEnableVertexAttribArray(0); // position
			glEnableVertexAttribArray(1); // normal
			glEnableVertexAttribArray(2); // tex coords.
			glEnableVertexAttribArray(3); // tangent.
			glEnableVertexAttribArray(4); // bitangent.
			glEnableVertexAttribArray(5); // normal.
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, gb.m_elementSizeInBytes, 0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, gb.m_elementSizeInBytes, (GLvoid*)(3 * sizeof(float)));
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, gb.m_elementSizeInBytes, (GLvoid*)(6 * sizeof(float)));
			glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, gb.m_elementSizeInBytes, (GLvoid*)(9 * sizeof(float)));
			glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, gb.m_elementSizeInBytes, (GLvoid*)(12 * sizeof(float)));
			glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, gb.m_elementSizeInBytes, (GLvoid*)(15 * sizeof(float)));

			initBuffer(tmpBuffer, gb.m_data, gb.m_sizeInBytes, GL_ARRAY_BUFFER, GL_STATIC_DRAW);
			m_buffers.push_back(tmpBuffer);

		//Unbind
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		SAFE_DELETE(geometry);

		m_isInitialized = true;
	}
};

#endif