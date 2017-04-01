////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	generalLoader.h
//
// summary:	Declaration and implementation of GeneralLoader class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __GENERALLOADER_H_
#define __GENERALLOADER_H_

#include <map>
#include <glm.hpp>
#include <dynArray.h>
#include "model.h"

using namespace std;

struct cmp_str : public std::binary_function<const char*, const char*, bool>
{
	bool operator()(char const *a, char const *b) const
	{
	    return std::strcmp(a, b) < 0;
	}
};

class GeneralLoader 
{
protected:
	map<const char*, int, cmp_str>* m_materialMapping;

public:
	void computeNormals(Model *model);

	virtual Model* loadModel(char* fileName){ return 0; }

protected:
	bool sameArrays(float* arr1, float* arr2, int count);
	void createDefaultMaterial(Model* model);
	int containsSameMaterial(Model* model, Material *material);
	Material* copyMaterial(Material &material);
	int findMaterial(Model *model, const char* matName);

	virtual void loadMaterial(char* fileName, Model* model){}
};


inline void GeneralLoader::createDefaultMaterial(Model* model)
{
	if (model->m_materials == NULL)
		model->m_materials = new DynArray<Material>(4);

	Material mat;

	Material *nm = copyMaterial(mat);
	model->m_materials->push_back(*nm);
	strcpy_s(nm->m_name, "Default");
	m_materialMapping->insert(std::pair<char *, int>(nm->m_name, 0));
}

inline bool GeneralLoader::sameArrays(float* arr1, float* arr2, int count)
{
	for (int i = 0; i<count; i++)
	{
		if (arr1[i] != arr2[i])
			return false;
	}
	return true;
}

inline int GeneralLoader::containsSameMaterial(Model* model, Material *material)
{
	Material *iter = model->m_materials->front();
	int i = 0;
	for (unsigned int i = 0; i< model->m_materials->size(); i++, iter++)
	{
		if (*material == *iter) return i;
	}
	return -1;
}

inline Material* GeneralLoader::copyMaterial(Material &material)
{
	Material *mat = new Material();
	strcpy_s(mat->m_name, material.m_name);
	mat->m_ambient[0] = material.m_ambient[0];
	mat->m_ambient[1] = material.m_ambient[1];
	mat->m_ambient[2] = material.m_ambient[2];
	mat->m_ambient[3] = material.m_ambient[3];

	mat->m_diffuse[0] = material.m_diffuse[0];
	mat->m_diffuse[1] = material.m_diffuse[1];
	mat->m_diffuse[2] = material.m_diffuse[2];
	mat->m_diffuse[3] = material.m_diffuse[3];

	mat->m_specular[0] = material.m_specular[0];
	mat->m_specular[1] = material.m_specular[1];
	mat->m_specular[2] = material.m_specular[2];
	mat->m_specular[3] = material.m_specular[3];

	strcpy_s(mat->m_diffuseTexture, material.m_diffuseTexture);

	mat->m_transparency = material.m_transparency;
	mat->m_shininess = material.m_shininess;
	mat->m_illumination = material.m_illumination;
	return mat;
}

inline int GeneralLoader::findMaterial(Model *model, const char* matName)
{
	map<const char*, int, cmp_str>::const_iterator iterMap;
	iterMap = m_materialMapping->find(matName);
	if (iterMap == m_materialMapping->end())
		return -1;//throw "Material not found";
	return iterMap->second;
}

inline void GeneralLoader::computeNormals(Model *model)
{
	// for each point
	//for (unsigned int pi=0; pi<model->points->size(); pi++)
	//{
	//	// point
	//	glm::vec3 &p = model->points->at(pi);
	//	// point normal
	//	glm::vec3 n(0,0,0);
	//	int count = 0;
	//	// find all faces with this point
	//	for(unsigned int mpi=0; mpi<model->modelParts->size(); mpi++)
	//	{
	//		ModelPart *mp = model->modelParts->at(mpi);
	//		for(unsigned int ii=0; ii<mp->vIds[GFT_TRIANGLE].size(); ii++)
	//		{
	//			if (mp->vIds[GFT_TRIANGLE].at(ii)==pi)
	//			{
	//				//n += mp->facesNormals[GFT_TRIANGLE].at(ii / 3);
	//				count++;
	//			}
	//		}
	//	}
	//	n /= count;
	//	n = glm::normalize(n);
	//	model->normals->push_back(n);
	//}
}


#endif