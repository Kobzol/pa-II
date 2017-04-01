////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	eai.h
//
// summary:	Declaration and implementation of ELEMENT ARRAY INFO class ... Says how VAO will be drown
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __EAI_H_
#define __EAI_H_

struct ELEMENT_ARRAY_INFO
{
	GLuint m_glGeometry;
	unsigned int m_vertexCount;		//The number of vertices per geometry, e.g. 3 for GL_TRIANGLES, 4 for GL_QUADS
	unsigned int m_noIndices;		//The number of relevant indices in the geometryBuffer, e.g. glGeometry = GL_TRIANGLES, then if count = 12  =>  we will draw 4 triangles
	int m_materialID;				//Material ID
	int m_startIndex;				//StartIndex used in glDrawArrays 

	inline void setGeometry(unsigned int eCount)
	{
		switch (eCount)
		{
		case 3:	m_glGeometry = GL_TRIANGLES; break;
		case 4:	m_glGeometry = GL_QUADS; break;
		default: m_glGeometry = GL_TRIANGLES; break;
		}
	}

	inline void print()
	{
		printf("EAI\n");
		printf("\tgeometry: %s\n", m_vertexCount == 3 ? "GL_TRIANGLES" : "GL_QUADS");
		printf("\tvCount: %u\n", m_vertexCount);
		printf("\tcount: %u\n", m_noIndices);
		printf("\tmId: %d\n", m_materialID);
		printf("\tstartIndex: %d\n", m_startIndex);
	}
};

#endif