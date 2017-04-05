////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	model.h
//
// summary:	Declaration and implementation of Model class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __MODEL_H_
#define __MODEL_H_

#include <stdlib.h>
#include <stdio.h>

#include <glm.hpp>
#include <tangentSpace.h>

#include <dynArray.h>
#include <buffer.h>

#include "boundingBox.h"
#include "material.h"
#include "modelPart.h"

#include "eai.h"

#define VERTEX_PNT_SIZE_IN_BYTES  36					// 3fv position | 3fv normal | 3fvTexCoords = (3 + 3 + 3) * sizeof(float)
#define VERTEX_PNT_T_SIZE_IN_BYTES  52				// 3fv position | 3fv normal | 3fvTexCoords | 4fv tangent (3 + 3 + 3 + 4) * sizeof(float)
#define VERTEX_PNT_TBN_SIZE_IN_BYTES  72				// 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal = (3 + 3 + 3 + 3 + 3 + 3) * sizeof(float)
#define VERTEX_PNTP_SIZE_IN_BYTES 48

typedef struct PNT { float data[9]; } PNT;
typedef struct PNT_T { float data[13]; } PNT_T;
typedef struct PNT_TBN { float data[18]; } PNT_TBN;
typedef struct PNTP { float data[12]; } PNTP;

class Model
{
public:
	BoundingBox m_bbox;

	//Fields
	DynArray< glm::vec3 > *m_points;
	DynArray< glm::vec3 > *m_normals;
	DynArray< glm::vec3 > *m_texCoords;

	DynArray< Material > *m_materials;
	DynArray< ModelPart* > *m_modelParts;

	//Constructors
	Model() : m_points(0), m_normals(0), m_texCoords(0), m_materials(0), m_modelParts(0) {}
	~Model(void);	

public:
	//Public methods
	void allocatePoints(int count){m_points->resize(count);}
	void mergeModelParts();
	void prepareColors();
	void setColors(const unsigned int *indices, const unsigned int count, const glm::vec4 &color);

	void clearAll();
	void clearPoints();
	void clearMaterials();
	void clearModelParts();

	void print(const char *fileName) const;

	bool createVAOGeometry(DynArray<Buffer<PNT>> *&g, DynArray<ELEMENT_ARRAY_INFO> *&eai, const bool mergeParts);
	bool createVAOGeometryInterpolated(DynArray<Buffer<PNTP>> *&g, DynArray<ELEMENT_ARRAY_INFO> *&eai, const bool mergeParts, Model* other);
	bool createVAOGeometryWithT(DynArray<Buffer<PNT_T>> *&g, DynArray<ELEMENT_ARRAY_INFO> *&eai, const bool mergeParts);
	bool createVAOGeometryWithTBN(DynArray<Buffer<PNT_TBN>> *&g, DynArray<ELEMENT_ARRAY_INFO> *&eai, const bool mergeParts);

	bool updateT(DynArray<Buffer<PNT_T>> *g, DynArray<ELEMENT_ARRAY_INFO> *eai);
	bool updateTBN(DynArray<Buffer<PNT_TBN>> *g, DynArray<ELEMENT_ARRAY_INFO> *eai, bool inverseTBN);
};

inline void Model::clearPoints()
{
	SAFE_DELETE(m_points);
	SAFE_DELETE(m_normals);
	SAFE_DELETE(m_texCoords);
}

inline void Model::clearMaterials()
{
	SAFE_DELETE(m_materials);
}

inline void Model::clearModelParts()
{
	if (m_modelParts)
	{
		for(unsigned int i=0; i< m_modelParts->size(); i++)
		{
			delete m_modelParts->at(i);
		}
		SAFE_DELETE(m_modelParts);
	}
}
	
inline void Model::clearAll()
{
	clearPoints();
	clearModelParts();
}

inline void Model::mergeModelParts()
{
	if (m_modelParts)
	{
		if (m_modelParts->size() > 1)
		{
			for(unsigned int i=m_modelParts->size()-1; i>=1; i--)
			{
				m_modelParts->at(0)->appendData(m_modelParts->at(i));
				delete m_modelParts->at(i);
				m_modelParts->pop_back();
			}
		}
	}
}

inline Model::~Model(void) 
{
	clearAll();
}

inline void Model::print(const char *fileName) const
{
	FILE *f;
	fopen_s(&f, fileName, "wt");

	for (unsigned int i=0; i<m_points->size(); i++)
	{
		fprintf_s(f, "v %f %f %f\n", (*m_points)[i].x, (*m_points)[i].y, (*m_points)[i].z);
	}

	fflush(f);
	fclose(f);
}

inline bool Model::createVAOGeometry(DynArray<Buffer<PNT>> *&g, DynArray<ELEMENT_ARRAY_INFO> *&eai, const bool mergeParts)
{
	g = new DynArray<Buffer<PNT>>(4);

	//DynArray<ELEMENT_ARRAY_INFO> eai(16);
	eai = new DynArray<ELEMENT_ARRAY_INFO>(16);
	
	Buffer<PNT> gb;

	ModelPart *mpPtr;
	DynArray< ObjFace > *fPtr;
	DynArray< glm::vec3 > *fnPtr;
	ObjFace *face;

	float* dstPtr;
	float *srcPtr;
	int index;

	float defaultNormal[3] = {0.0f, 0.0f, 1.0f};
	float defaultTexCoords[3] = {0.0f, 0.0f, 0.0f};

	unsigned int vertexSizeInBytes =  VERTEX_PNT_SIZE_IN_BYTES;		//3 position floats + 3 normal vector floats + 3 texture coords. floats

	if (mergeParts)
		mergeModelParts();

	//Convert all model parts to geometry buffers such that a single VAO Vertex contains all needed attributes = position | normal | tex. coords
	for(unsigned int i=0; i<m_modelParts->size(); i++)					//Go through all mesh parts
	{
		mpPtr = m_modelParts->at(i);
		fPtr = &(mpPtr->faces);
		fnPtr = &(mpPtr->facesNormals);
			
		mpPtr->sortByMaterialAndGeometry();
	
		gb.m_elementSizeInBytes = vertexSizeInBytes;
		gb.m_noElements = mpPtr->noTriangles * 3 + mpPtr->noQuads * 4;
		gb.m_sizeInBytes = gb.m_noElements * gb.m_elementSizeInBytes;
		gb.m_data = (PNT*)malloc(gb.m_sizeInBytes);
		dstPtr = (float*)gb.m_data;

		face = fPtr->front();

		ELEMENT_ARRAY_INFO tmpEAI;
		tmpEAI.m_vertexCount = face->m_noVertices;
		tmpEAI.setGeometry(tmpEAI.m_vertexCount);
		tmpEAI.m_materialID = face->m_materiaID;
		tmpEAI.m_noIndices = 0;
		tmpEAI.m_startIndex = 0;

		for (unsigned int j=0; j<fPtr->size(); j++)						//Go through all faces of the mesh part 
		{
			if ((face->m_noVertices != tmpEAI.m_vertexCount) || (face->m_materiaID != tmpEAI.m_materialID))
			{
				eai->push_back(tmpEAI);
				tmpEAI.m_vertexCount = face->m_noVertices;
				tmpEAI.setGeometry(tmpEAI.m_vertexCount);
				tmpEAI.m_materialID = face->m_materiaID;
				tmpEAI.m_startIndex = tmpEAI.m_startIndex + tmpEAI.m_noIndices;
				tmpEAI.m_noIndices = 0;
			}
			tmpEAI.m_noIndices+=tmpEAI.m_vertexCount;

			for (unsigned int x=0; x<face->m_noVertices; x++)
			{
				index = face->m_vIds[x];																									
				srcPtr = &(*m_points)[index][0];							//data pointer to the glm::vec3 vertex coordinates
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//x																
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//y
				*dstPtr = *srcPtr;	dstPtr++;							//z
	   
				index = face->m_nIds[x];																									
				srcPtr = (index == -1) ? defaultNormal : &(*m_normals)[index][0];	//data pointer to the glm::vec3 normal coordinates
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//x
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//y
				*dstPtr = *srcPtr;	dstPtr++;							//z

				index = face->m_tIds[x];																									
				srcPtr = (index == -1) ? defaultTexCoords : &(*m_texCoords)[index][0];	//data pointer to the glm::vec3 texture coordinates		
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//x
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//y
				*dstPtr = *srcPtr;	dstPtr++;							//z
			}

			face++;
		}

		face--;
		eai->push_back(tmpEAI);
		g->push_back(gb);	//call copy constructor avoid deleting inner gb.m_data buffer

	}

	//Merge all geometry buffers  to be able to create a single "GL_STATIC_DRAW" VAO
	if (g->size() > 1)
	{
		unsigned int totalSizeInBytes = 0;
		for(unsigned int i=0; i<g->size();i++)
			totalSizeInBytes += g->at(i).m_sizeInBytes;

		gb.m_sizeInBytes = totalSizeInBytes;
		gb.m_elementSizeInBytes = vertexSizeInBytes;
		gb.m_noElements = gb.m_sizeInBytes / gb.m_elementSizeInBytes;
		gb.m_data = (PNT*)malloc(gb.m_sizeInBytes);
		char *dst = (char*)gb.m_data;
		for(unsigned int i=0; i<g->size();i++)
		{
			memcpy_s(dst, gb.m_sizeInBytes, g->at(i).m_data, g->at(i).m_sizeInBytes);
			dst+=g->at(i).m_sizeInBytes;
			g->at(i).~Buffer();
		}
		g->clear();
		g->push_back(gb);	//call copy constructor avoid deleting inner gb.m_data buffer
	}

	for(unsigned int i=1;i<eai->size(); i++)
	{
		(*eai)[i].m_startIndex = (*eai)[i-1].m_startIndex + (*eai)[i-1].m_noIndices;
	}

	return true;
}
inline bool Model::createVAOGeometryInterpolated(DynArray<Buffer<PNTP>> *&g, DynArray<ELEMENT_ARRAY_INFO> *&eai, const bool mergeParts, Model* other)
{
	g = new DynArray<Buffer<PNTP>>(4);

	//DynArray<ELEMENT_ARRAY_INFO> eai(16);
	eai = new DynArray<ELEMENT_ARRAY_INFO>(16);

	Buffer<PNTP> gb;

	ModelPart *mpPtr;
	DynArray< ObjFace > *fPtr;
	DynArray< glm::vec3 > *fnPtr;
	ObjFace *face;

	float* dstPtr;
	float *srcPtr;
	int index;

	ObjFace* otherFace;
	ModelPart* otherMpPtr;
	DynArray< ObjFace > *otherFPtr;

	float defaultNormal[3] = { 0.0f, 0.0f, 1.0f };
	float defaultTexCoords[3] = { 0.0f, 0.0f, 0.0f };

	unsigned int vertexSizeInBytes = VERTEX_PNTP_SIZE_IN_BYTES;		//3 position floats + 3 normal vector floats + 3 texture coords. floats

	if (mergeParts)
		mergeModelParts();

	//Convert all model parts to geometry buffers such that a single VAO Vertex contains all needed attributes = position | normal | tex. coords
	for (unsigned int i = 0; i<m_modelParts->size(); i++)					//Go through all mesh parts
	{
		mpPtr = m_modelParts->at(i);
		otherMpPtr = other->m_modelParts->at(i);
		otherFPtr = &(otherMpPtr->faces);
		
		fPtr = &(mpPtr->faces);
		fnPtr = &(mpPtr->facesNormals);

		mpPtr->sortByMaterialAndGeometry();
		otherMpPtr->sortByMaterialAndGeometry();

		gb.m_elementSizeInBytes = vertexSizeInBytes;
		gb.m_noElements = mpPtr->noTriangles * 3 + mpPtr->noQuads * 4;
		gb.m_sizeInBytes = gb.m_noElements * gb.m_elementSizeInBytes;
		gb.m_data = (PNTP*)malloc(gb.m_sizeInBytes);
		dstPtr = (float*)gb.m_data;

		face = fPtr->front();
		otherFace = otherFPtr->front();

		ELEMENT_ARRAY_INFO tmpEAI;
		tmpEAI.m_vertexCount = face->m_noVertices;
		tmpEAI.setGeometry(tmpEAI.m_vertexCount);
		tmpEAI.m_materialID = face->m_materiaID;
		tmpEAI.m_noIndices = 0;
		tmpEAI.m_startIndex = 0;

		for (unsigned int j = 0; j<fPtr->size(); j++)						//Go through all faces of the mesh part 
		{
			if ((face->m_noVertices != tmpEAI.m_vertexCount) || (face->m_materiaID != tmpEAI.m_materialID))
			{
				eai->push_back(tmpEAI);
				tmpEAI.m_vertexCount = face->m_noVertices;
				tmpEAI.setGeometry(tmpEAI.m_vertexCount);
				tmpEAI.m_materialID = face->m_materiaID;
				tmpEAI.m_startIndex = tmpEAI.m_startIndex + tmpEAI.m_noIndices;
				tmpEAI.m_noIndices = 0;
			}
			tmpEAI.m_noIndices += tmpEAI.m_vertexCount;

			for (unsigned int x = 0; x<face->m_noVertices; x++)
			{
				index = face->m_vIds[x];
				srcPtr = &(*m_points)[index][0];							//data pointer to the glm::vec3 vertex coordinates
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//x																
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//y
				*dstPtr = *srcPtr;	dstPtr++;							//z

				index = face->m_nIds[x];
				srcPtr = (index == -1) ? defaultNormal : &(*m_normals)[index][0];	//data pointer to the glm::vec3 normal coordinates
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//x
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//y
				*dstPtr = *srcPtr;	dstPtr++;							//z

				index = face->m_tIds[x];
				srcPtr = (index == -1) ? defaultTexCoords : &(*m_texCoords)[index][0];	//data pointer to the glm::vec3 texture coordinates		
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//x
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//y
				*dstPtr = *srcPtr;	dstPtr++;							//z

				srcPtr = &(*other->m_points)[face->m_vIds[x]][0];
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//x																
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//y
				*dstPtr = *srcPtr;	dstPtr++;							//z
			}

			face++;
		}

		face--;
		eai->push_back(tmpEAI);
		g->push_back(gb);	//call copy constructor avoid deleting inner gb.m_data buffer

	}

	//Merge all geometry buffers  to be able to create a single "GL_STATIC_DRAW" VAO
	if (g->size() > 1)
	{
		unsigned int totalSizeInBytes = 0;
		for (unsigned int i = 0; i<g->size(); i++)
			totalSizeInBytes += g->at(i).m_sizeInBytes;

		gb.m_sizeInBytes = totalSizeInBytes;
		gb.m_elementSizeInBytes = vertexSizeInBytes;
		gb.m_noElements = gb.m_sizeInBytes / gb.m_elementSizeInBytes;
		gb.m_data = (PNTP*)malloc(gb.m_sizeInBytes);
		char *dst = (char*)gb.m_data;
		for (unsigned int i = 0; i<g->size(); i++)
		{
			memcpy_s(dst, gb.m_sizeInBytes, g->at(i).m_data, g->at(i).m_sizeInBytes);
			dst += g->at(i).m_sizeInBytes;
			g->at(i).~Buffer();
		}
		g->clear();
		g->push_back(gb);	//call copy constructor avoid deleting inner gb.m_data buffer
	}

	for (unsigned int i = 1; i<eai->size(); i++)
	{
		(*eai)[i].m_startIndex = (*eai)[i - 1].m_startIndex + (*eai)[i - 1].m_noIndices;
	}

	return true;
}

inline bool Model::createVAOGeometryWithT(DynArray<Buffer<PNT_T>> *&g, DynArray<ELEMENT_ARRAY_INFO> *&eai, const bool mergeParts)
{
	g = new DynArray<Buffer<PNT_T>>(4);

	//DynArray<ELEMENT_ARRAY_INFO> eai(16);
	eai = new DynArray<ELEMENT_ARRAY_INFO>(16);
	
	Buffer<PNT_T> gb;

	ModelPart *mpPtr;
	DynArray< ObjFace > *fPtr;
	DynArray< glm::vec3 > *fnPtr;
	ObjFace *face;

	float* dstPtr;
	float *srcPtr;
	int index;

	float defaultNormal[3] = {0.0f, 0.0f, 1.0f};
	float defaultTexCoords[3] = {0.0f, 0.0f, 0.0f};

	unsigned int vertexSizeInBytes =  VERTEX_PNT_T_SIZE_IN_BYTES;		//3 position floats + 3 normal vector floats + 3 texture coords. floats

	if (mergeParts)
		mergeModelParts();

	//Convert all model parts to geometry buffers such that a single VAO Vertex contains all needed attributes = position | normal | tex. coords
	for(unsigned int i=0; i<m_modelParts->size(); i++)					//Go through all mesh parts
	{
		mpPtr = m_modelParts->at(i);
		fPtr = &(mpPtr->faces);
		fnPtr = &(mpPtr->facesNormals);
			
		mpPtr->sortByMaterialAndGeometry();
	
		gb.m_elementSizeInBytes = vertexSizeInBytes;
		gb.m_noElements = mpPtr->noTriangles * 3 + mpPtr->noQuads * 4;
		gb.m_sizeInBytes = gb.m_noElements * gb.m_elementSizeInBytes;
		gb.m_data = (PNT_T*)malloc(gb.m_sizeInBytes);
		dstPtr = (float*)gb.m_data;

		face = fPtr->front();

		ELEMENT_ARRAY_INFO tmpEAI;
		tmpEAI.m_vertexCount = face->m_noVertices;
		tmpEAI.setGeometry(tmpEAI.m_vertexCount);
		tmpEAI.m_materialID = face->m_materiaID;
		tmpEAI.m_noIndices = 0;
		tmpEAI.m_startIndex = 0;

		for (unsigned int j=0; j<fPtr->size(); j++)						//Go through all faces of the mesh part 
		{
			if ((face->m_noVertices != tmpEAI.m_vertexCount) || (face->m_materiaID != tmpEAI.m_materialID))
			{
				eai->push_back(tmpEAI);
				tmpEAI.m_vertexCount = face->m_noVertices;
				tmpEAI.setGeometry(tmpEAI.m_vertexCount);
				tmpEAI.m_materialID = face->m_materiaID;
				tmpEAI.m_startIndex = tmpEAI.m_startIndex + tmpEAI.m_noIndices;
				tmpEAI.m_noIndices = 0;
			}
			tmpEAI.m_noIndices+=tmpEAI.m_vertexCount;

			for (unsigned int x=0; x<face->m_noVertices; x++)
			{
				index = face->m_vIds[x];																									
				srcPtr = &(*m_points)[index][0];							//data pointer to the glm::vec3 vertex coordinates
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//x																
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//y
				*dstPtr = *srcPtr;	dstPtr++;							//z
	   
				index = face->m_nIds[x];																									
				srcPtr = (index == -1) ? defaultNormal : &(*m_normals)[index][0];	//data pointer to the glm::vec3 normal coordinates
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//x
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//y
				*dstPtr = *srcPtr;	dstPtr++;							//z

				index = face->m_tIds[x];																									
				srcPtr = (index == -1) ? defaultTexCoords : &(*m_texCoords)[index][0];	//data pointer to the glm::vec3 texture coordinates		
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//x
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//y
				*dstPtr = *srcPtr;	dstPtr++;							//z

				dstPtr += 4;											//skip tangent = 4fv
			}

			face++;
		}

		face--;
		eai->push_back(tmpEAI);
			
		g->push_back(gb);
	}

	//Merge all geometry buffers  to be able to create a single "GL_STATIC_DRAW" VAO
	if (g->size() > 1)
	{
		unsigned int totalSizeInBytes = 0;
		for(unsigned int i=0; i<g->size();i++)
			totalSizeInBytes += g->at(i).m_sizeInBytes;

		gb.m_sizeInBytes = totalSizeInBytes;
		gb.m_elementSizeInBytes = vertexSizeInBytes;
		gb.m_noElements = gb.m_sizeInBytes / gb.m_elementSizeInBytes;
		gb.m_data = (PNT_T*)malloc(gb.m_sizeInBytes);
		char *dst = (char*)gb.m_data;
		for(unsigned int i=0; i<g->size();i++)
		{
			memcpy_s(dst, gb.m_sizeInBytes, g->at(i).m_data, g->at(i).m_sizeInBytes);
			dst += g->at(i).m_sizeInBytes;
			g->at(i).~Buffer();
		}
		g->clear();
		g->push_back(gb);
	}

	for(unsigned int i=1;i<eai->size(); i++)
	{
		(*eai)[i].m_startIndex = (*eai)[i-1].m_startIndex + (*eai)[i-1].m_noIndices;
	}

	return true;
}

inline bool Model::updateT(DynArray<Buffer<PNT_T>> *g, DynArray<ELEMENT_ARRAY_INFO> *eai)
{
	if ((eai == 0) || (g==0)) return false;

	//TangentSpace ts;

	//glm::vec3* dstPtr;
	
	/*
	unsigned int id0, id1, id2;
	glm::vec3* srcPtr;
	glm::vec3* srcPtrV0;
	glm::vec3* srcPtrV1;
	glm::vec3* srcPtrV2;

	for(unsigned int i=0;i<eai->size(); i++)
	{
		srcPtr = (glm::vec3*)g->dataBuffers[i].data;

		if (eai->at(i).vCount == 4)
		{
			for (unsigned int j=0; j<eai->at(i).count; j+=4)
			{
				//for (unsigned int k=0; k<4; k++)
				//{
				//	id0 = ((0+k)%4) * 6;			// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 
				//	id1 = ((1+k)%4) * 6;			// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 
				//	id2 = ((3+k)%4) * 6;			// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 

				//	srcPtrV0 = srcPtr + id0;
				//	srcPtrV1 = srcPtr + id1;
				//	srcPtrV2 = srcPtr + id2;
				//	dstPtr = srcPtrV0 + 3;			// tangent vecotr of the id0 vertex
				//	ts.createNormalizedInverseTBN(srcPtrV0, srcPtrV1, srcPtrV2, srcPtrV0 + 2, srcPtrV1 + 2, srcPtrV2 + 2, dstPtr, dstPtr+1, dstPtr+2);
				//}

				//srcPtr += 4 * 6;					// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 
			}
		}

		if (eai->at(i).vCount == 3)
		{
			for (unsigned int j=0; j<eai->at(i).count; j+=3)
			{
				//for (unsigned int k=0; k<3; k++)
				//{
				//	id0 = ((0+k)%3) * 6;			// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 
				//	id1 = ((1+k)%3) * 6;			// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 
				//	id2 = ((2+k)%3) * 6;			// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 

				//	srcPtrV0 = srcPtr + id0;
				//	srcPtrV1 = srcPtr + id1;
				//	srcPtrV2 = srcPtr + id2;

				//	dstPtr = srcPtrV0 + 3;			// tangent vector of the id0 vertex

				//	ts.createNormalizedInverseTBN(srcPtrV0, srcPtrV1, srcPtrV2, srcPtrV0 + 2, srcPtrV1 + 2, srcPtrV2 + 2, dstPtr, dstPtr+1, dstPtr+2);
				//}
				//srcPtr += 3 * 6;					// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 
			}
		}	
	}
	*/
}

inline bool Model::createVAOGeometryWithTBN(DynArray<Buffer<PNT_TBN>> *&g, DynArray<ELEMENT_ARRAY_INFO> *&eai, const bool mergeParts)
{
	g = new DynArray<Buffer<PNT_TBN>>(4);

	//DynArray<ELEMENT_ARRAY_INFO> eai(16);
	eai = new DynArray<ELEMENT_ARRAY_INFO>(16);
	
	Buffer<PNT_TBN> gb;

	ModelPart *mpPtr;
	DynArray< ObjFace > *fPtr;
	DynArray< glm::vec3 > *fnPtr;
	ObjFace *face;

	float* dstPtr;
	float *srcPtr;
	int index;

	float defaultNormal[3] = {0.0f, 0.0f, 1.0f};
	float defaultTexCoords[3] = {0.0f, 0.0f, 0.0f};

	unsigned int vertexSizeInBytes =  VERTEX_PNT_TBN_SIZE_IN_BYTES;		//3 position floats + 3 normal vector floats + 3 texture coords. floats

	if (mergeParts)
		mergeModelParts();

	//Convert all model parts to geometry buffers such that a single VAO Vertex contains all needed attributes = position | normal | tex. coords
	for(unsigned int i=0; i<m_modelParts->size(); i++)					//Go through all mesh parts
	{
		mpPtr = m_modelParts->at(i);
		fPtr = &(mpPtr->faces);
		fnPtr = &(mpPtr->facesNormals);
			
		mpPtr->sortByMaterialAndGeometry();
	
		gb.m_elementSizeInBytes = vertexSizeInBytes;
		gb.m_noElements = mpPtr->noTriangles * 3 + mpPtr->noQuads * 4;
		gb.m_sizeInBytes = gb.m_noElements * gb.m_elementSizeInBytes;
		gb.m_data = (PNT_TBN*)malloc(gb.m_sizeInBytes);
		dstPtr = (float*)gb.m_data;

		face = fPtr->front();

		ELEMENT_ARRAY_INFO tmpEAI;
		tmpEAI.m_vertexCount = face->m_noVertices;
		tmpEAI.setGeometry(tmpEAI.m_vertexCount);
		tmpEAI.m_materialID = face->m_materiaID;
		tmpEAI.m_noIndices = 0;
		tmpEAI.m_startIndex = 0;

		for (unsigned int j=0; j<fPtr->size(); j++)						//Go through all faces of the mesh part 
		{
			if ((face->m_noVertices != tmpEAI.m_vertexCount) || (face->m_materiaID != tmpEAI.m_materialID))
			{
				eai->push_back(tmpEAI);
				tmpEAI.m_vertexCount = face->m_noVertices;
				tmpEAI.setGeometry(tmpEAI.m_vertexCount);
				tmpEAI.m_materialID = face->m_materiaID;
				tmpEAI.m_startIndex = tmpEAI.m_startIndex + tmpEAI.m_noIndices;
				tmpEAI.m_noIndices = 0;
			}
			tmpEAI.m_noIndices+=tmpEAI.m_vertexCount;

			for (unsigned int x=0; x<face->m_noVertices; x++)
			{
				index = face->m_vIds[x];																									
				srcPtr = &(*m_points)[index][0];							//data pointer to the glm::vec3 vertex coordinates
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//x																
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//y
				*dstPtr = *srcPtr;	dstPtr++;							//z
	   
				index = face->m_nIds[x];																									
				srcPtr = (index == -1) ? defaultNormal : &(*m_normals)[index][0];	//data pointer to the glm::vec3 normal coordinates
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//x
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//y
				*dstPtr = *srcPtr;	dstPtr++;							//z

				index = face->m_tIds[x];																									
				srcPtr = (index == -1) ? defaultTexCoords : &(*m_texCoords)[index][0];	//data pointer to the glm::vec3 texture coordinates		
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//x
				*dstPtr = *srcPtr;	dstPtr++; srcPtr++;					//y
				*dstPtr = *srcPtr;	dstPtr++;							//z

				dstPtr += 9;											//skip tangent, binormal, normal vector = 3 * 3fv
			}

			face++;
		}

		face--;
		eai->push_back(tmpEAI);
			
		g->push_back(gb);
	}

	//Merge all geometry buffers  to be able to create a single "GL_STATIC_DRAW" VAO
	if (g->size() > 1)
	{
		unsigned int totalSizeInBytes = 0;
		for(unsigned int i=0; i<g->size();i++)
			totalSizeInBytes += g->at(i).m_sizeInBytes;

		gb.m_sizeInBytes = totalSizeInBytes;
		gb.m_elementSizeInBytes = vertexSizeInBytes;
		gb.m_noElements = gb.m_sizeInBytes / gb.m_elementSizeInBytes;
		gb.m_data = (PNT_TBN*)malloc(gb.m_sizeInBytes);
		char *dst = (char*)gb.m_data;
		for(unsigned int i=0; i<g->size();i++)
		{
			memcpy_s(dst, gb.m_sizeInBytes, g->at(i).m_data, g->at(i).m_sizeInBytes);
			dst += g->at(i).m_sizeInBytes;
			g->at(i).~Buffer();
		}
		g->clear();
		g->push_back(gb);
	}

	for(unsigned int i=1;i<eai->size(); i++)
	{
		(*eai)[i].m_startIndex = (*eai)[i-1].m_startIndex + (*eai)[i-1].m_noIndices;
	}

	return true;
}

inline bool Model::updateTBN(DynArray<Buffer<PNT_TBN>> *g, DynArray<ELEMENT_ARRAY_INFO> *eai, bool inverseTBN)
{
	if ((eai == 0) || (g==0)) return false;

	TangentSpace ts;

	//Buffer gb;
	glm::vec3* dstPtr;
	
	unsigned int id0, id1, id2;
	glm::vec3* srcPtr;
	glm::vec3* srcPtrV0;
	glm::vec3* srcPtrV1;
	glm::vec3* srcPtrV2;

	for(unsigned int i=0;i<eai->size(); i++)
	{
		srcPtr = (glm::vec3*)g->at(i).m_data;

		if (eai->at(i).m_vertexCount == 4)
		{
			for (unsigned int j=0; j<eai->at(i).m_noIndices; j+=4)
			{
				for (unsigned int k=0; k<4; k++)
				{
					id0 = ((0+k)%4) * 6;			// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 
					id1 = ((1+k)%4) * 6;			// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 
					id2 = ((3+k)%4) * 6;			// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 

					srcPtrV0 = srcPtr + id0;
					srcPtrV1 = srcPtr + id1;
					srcPtrV2 = srcPtr + id2;
					dstPtr = srcPtrV0 + 3;			// tangent vecotr of the id0 vertex
					if (inverseTBN)
						ts.createNormalizedInverseTBN(srcPtrV0, srcPtrV1, srcPtrV2, srcPtrV0 + 2, srcPtrV1 + 2, srcPtrV2 + 2, dstPtr, dstPtr+1, dstPtr+2);
					else
						ts.createNormalizedTBN(srcPtrV0, srcPtrV1, srcPtrV2, srcPtrV0 + 2, srcPtrV1 + 2, srcPtrV2 + 2, dstPtr, dstPtr+1, dstPtr+2);
				}

				srcPtr += 4 * 6;					// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 
			}
		}

		if (eai->at(i).m_vertexCount == 3)
		{
			for (unsigned int j=0; j<eai->at(i).m_noIndices; j+=3)
			{
				for (unsigned int k=0; k<3; k++)
				{
					id0 = ((0+k)%3) * 6;			// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 
					id1 = ((1+k)%3) * 6;			// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 
					id2 = ((2+k)%3) * 6;			// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 

					srcPtrV0 = srcPtr + id0;
					srcPtrV1 = srcPtr + id1;
					srcPtrV2 = srcPtr + id2;

					dstPtr = srcPtrV0 + 3;			// tangent vector of the id0 vertex
					
					if (inverseTBN)
						ts.createNormalizedInverseTBN(srcPtrV0, srcPtrV1, srcPtrV2, srcPtrV0 + 2, srcPtrV1 + 2, srcPtrV2 + 2, dstPtr, dstPtr+1, dstPtr+2);
					else
						ts.createNormalizedTBN(srcPtrV0, srcPtrV1, srcPtrV2, srcPtrV0 + 2, srcPtrV1 + 2, srcPtrV2 + 2, dstPtr, dstPtr+1, dstPtr+2);
				}
				srcPtr += 3 * 6;					// * 6 skips 3fv position | 3fv normal | 3fvTexCoords | 3fv tangent | 3fv binormal | 3fv normal 
			}
		}
	}	
	return true;
}

#endif
