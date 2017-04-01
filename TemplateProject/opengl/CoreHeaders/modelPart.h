////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	modelPart.h
//
// summary:	Declaration and implementation of ModelPart class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __MODELPART_H_
#define __MODELPART_H_

#include <dynArray.h>
#include <glm.hpp>
#include <sorting.h>

#define MAX_VERTEX_COUNT 4 //can only handle quads or triangles

enum FaceFlag SMALLSIZE
{
	FF_NONE = 0,
	FF_HAS_NID	= 1,
	FF_HAS_TID = 2,
	FF_HAS_ALL = FF_HAS_NID | FF_HAS_TID
};

struct ObjFace
{
	int m_vIds[MAX_VERTEX_COUNT];					//indices to the model points, vertices
	int m_nIds[MAX_VERTEX_COUNT];					//indices to the model normals
	int m_tIds[MAX_VERTEX_COUNT];					//indices to the model texture coordinates
	int m_materiaID;									//model material Id
	unsigned char m_noVertices;						//number of vertices
	FaceFlag m_flag;

	inline bool operator < (const ObjFace &o) const 
	{
        if (m_materiaID > o.m_materiaID) return false;
		if (m_materiaID == o.m_materiaID) return m_noVertices < o.m_noVertices;
		return true;
    }

	inline bool operator > (const ObjFace &o) const 
	{
        if (m_materiaID < o.m_materiaID) return false;
		if (m_materiaID == o.m_materiaID) return m_noVertices > o.m_noVertices;
		return true;
    }
};

class ModelPart
{
public:
	char objectName[256];
	bool useFaceNormals;
	unsigned int noTriangles;
	unsigned int noQuads;

	DynArray< ObjFace > faces;
	DynArray< glm::vec3 > facesNormals;		

public:
	ModelPart() : useFaceNormals(false), noTriangles(0), noQuads(0), faces(DynArray< ObjFace >(1<<18)), facesNormals(DynArray< glm::vec3 >(1<<18)) {}
	~ModelPart(void) {};

	void appendData(ModelPart *other)
	{
		faces.insert_at_end(other->faces.front(), other->faces.size());
		facesNormals.insert_at_end(other->facesNormals.front(), other->facesNormals.size());
	}

	void sortByMaterialAndGeometry()
	{
		Sorting<ObjFace, glm::vec3 >::quickSort(faces, facesNormals, 0, faces.size()-1);
	}
};

#endif