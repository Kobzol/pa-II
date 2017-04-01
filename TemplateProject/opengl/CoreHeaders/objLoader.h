////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	objLoader.h
//
// summary:	Declaration and implementation of OBJLoader class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __OBJLOADER_H_
#define __OBJLOADER_H_

#include <dynArray.h>
#include "model.h"
#include "generalLoader.h"

using namespace std;

class ObjLoader : public GeneralLoader
{
public:
	Model* loadModel(const char* fileName);

private:
	void loadMaterial(const char* fileName, Model* model);
};

inline Model* ObjLoader::loadModel(const char* fileName)
{
	char defaultMat[8];
	strcpy_s(defaultMat, 8, "Default");
	Model* model = new Model();
	m_materialMapping = new map<const char*, int, cmp_str>();

	model->m_points = new DynArray<glm::vec3>(1 << 20);
	model->m_normals = new DynArray<glm::vec3>(1 << 20);
	model->m_texCoords = new DynArray<glm::vec3>(1 << 20);
	model->m_modelParts = new DynArray<ModelPart*>(8);

	FILE *f = 0;
	fopen_s(&f, fileName, "r");
	if (f == NULL)
		throw "OBJ file not found";

	int lineLength = (1 << 20);
	char* line = (char*)malloc(sizeof(char) * lineLength);

	int matIndex = -1;
	ModelPart *part = NULL;

	char* vInfo[16];
	unsigned char vCount = 0;
	FaceFlag fFlag = FF_NONE;
	int *idx = 0;
	int *nid = 0;
	int *tid = 0;

	while (fgets(line, 1 << 20, f) != NULL)
	{
		// ignore comments
		if (line[0] == '#')
			continue;

		else if (strncmp(line, "o ", 2) == 0)
		{
			if (part != NULL)
				model->m_modelParts->push_back(part);
			part = new ModelPart();
			//strcpy_s(part->objectName, 256, &line[3]);	
			sscanf_s(&line[3], "%[^\n]", part->objectName, sizeof(part->objectName));
		}

		// load materials
		else if (strncmp(line, "mtllib", 6) == 0)
		{
			char buf[256];
			strcpy_s(buf, 256, fileName);
			char* ptr = strrchr(buf, '/');
			unsigned int restLength = (unsigned int)(ptr - buf + 1);
			strcpy_s(ptr + 1, 256 - restLength, &line[7]);
			buf[strlen(buf) - 1] = 0;
			loadMaterial(buf, model);
		}

		// vertices
		else if (strncmp(line, "v ", 2) == 0)
		{
			glm::vec3 v;
			sscanf_s(line, "v %f %f %f", &(v.x), &(v.y), &(v.z));
			//printf("%u: %f %f %f\n", model->points->size(), v.x, v.y, v.z);
			model->m_points->push_back(v);
		}

		// normals
		else if (strncmp(line, "vn", 2) == 0)
		{
			glm::vec3 vn;
			sscanf_s(line, "vn %f %f %f", &(vn.x), &(vn.y), &(vn.z));
			model->m_normals->push_back(vn);
		}

		// texture coords
		else if (strncmp(line, "vt", 2) == 0)
		{
			glm::vec3 vt;
			sscanf_s(line, "vt %f %f %f", &(vt.x), &(vt.y), &(vt.z));
			model->m_texCoords->push_back(vt);
		}

		// material
		else if (strncmp(line, "usemtl", 6) == 0)
		{
			char buf[256];
			//strcpy_s(buf, 256, &line[7]);	
			sscanf_s(&line[7], "%[^\n]", buf, 256);
			int newMat = findMaterial(model, buf);
			if (newMat == -1)
			{
				createDefaultMaterial(model);
				newMat = findMaterial(model, (char*)defaultMat);
			}
			if (newMat != matIndex)
			{
				matIndex = newMat;
			}
		}

		// faces
		else if (strncmp(line, "f", 1) == 0)
		{
			if (part == NULL)// && part->materialIndex==11)
			{
				part = new ModelPart();
				strcpy_s(part->objectName, 256, "Noname object");
			}

			//Get number of face vertices
			char* data = &line[2];
			vCount = 0;
			while (*data != 0)
			{
				if (*data == ' ' || *data == '\t' || *data == '\r' || *data == '\n')
				{
					data++;
					continue;
				}
				else
				{
					vInfo[vCount++] = data;
					while (*data != ' ' && *data != '\t' && *data != '\r' && *data != '\n' && *data != 0)
					{
						data++;
					}
				}
			}

			//Get face definition type
			fFlag = FF_NONE;
			data = vInfo[vCount - 1];
			while (*data != 0)
			{
				if (*data == '/')
					if ((fFlag & FF_HAS_TID) == 0)
						fFlag = (FaceFlag)(fFlag | FF_HAS_TID);
					else
						fFlag = (*(data - 1) == '/') ? FF_HAS_NID : (FaceFlag)(fFlag | FF_HAS_NID);
				data++;
			}

			//new Face added
			part->faces.push_back(ObjFace());
			ObjFace *tmpF = &(part->faces[part->faces.size() - 1]);
			idx = &(tmpF->m_vIds[0]);
			nid = &(tmpF->m_nIds[0]);
			tid = &(tmpF->m_tIds[0]);

			tmpF->m_noVertices = vCount;
			tmpF->m_materiaID = matIndex;
			tmpF->m_flag = fFlag;

			if (vCount == 3) part->noTriangles++;
			else if (vCount == 4) part->noQuads++;
			else break;											//This should never happend

			switch (fFlag)
			{
			case FF_NONE:
				for (unsigned int i = 0; i<vCount; i++)
				{
					sscanf_s(vInfo[i], "%d", &idx[i]);
					idx[i]--;									// must be decreased, obj index from 1, c++ from 0 
					tid[i] = -1;
					nid[i] = -1;
				}
				break;
			case FF_HAS_ALL:
				for (unsigned int i = 0; i<vCount; i++)
				{
					sscanf_s(vInfo[i], "%d/%d/%d", &idx[i], &tid[i], &nid[i]);
					idx[i]--;									// must be decreased, obj index from 1, c++ from 0 
					tid[i]--;									// must be decreased, obj index from 1, c++ from 0 
					nid[i]--;									// must be decreased, obj index from 1, c++ from 0 
				}
				break;
			case FF_HAS_TID:
				for (unsigned int i = 0; i<vCount; i++)
				{
					sscanf_s(vInfo[i], "%d/%d", &idx[i], &tid[i]);
					idx[i]--;									// must be decreased, obj index from 1, c++ from 0 
					tid[i]--;									// must be decreased, obj index from 1, c++ from 0 
					nid[i] = -1;
				}
				break;
			case FF_HAS_NID:
				for (unsigned int i = 0; i<vCount; i++)
				{
					sscanf_s(vInfo[i], "%d//%d", &idx[i], &nid[i]);
					idx[i]--;									// must be decreased, obj index from 1, c++ from 0 
					tid[i] = -1;
					nid[i]--;									// must be decreased, obj index from 1, c++ from 0 
				}
				break;
			default:
				break;
			}

			glm::vec3 &p0 = model->m_points->at(idx[0]);
			glm::vec3 &p1 = model->m_points->at(idx[1]);
			glm::vec3 &p2 = model->m_points->at(idx[2]);
			glm::vec3 v1 = p2 - p1;
			glm::vec3 v2 = p0 - p1;
			glm::vec3 n = glm::cross(v1, v2);
			n = (glm::length(n) <= 0.0001) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::normalize(n);
			part->facesNormals.push_back(n);
		}
	}

	if (part != NULL)
		model->m_modelParts->push_back(part);

	fclose(f);

	for (unsigned int i = 0; i<model->m_modelParts->size(); i++)
	{
		ModelPart *mp = model->m_modelParts->at(i);
		for (unsigned int j = 0; j<mp->faces.size(); j++)
		{
			int newMat = -1;

			if ((mp->faces[j].m_materiaID == -1) && ((newMat = findMaterial(model, (char*)defaultMat)) == -1))
			{
				createDefaultMaterial(model);
				newMat = findMaterial(model, defaultMat);
				mp->faces[j].m_materiaID = newMat;
			}
			else if (mp->faces[j].m_materiaID == -1)
			{
				mp->faces[j].m_materiaID = newMat;
			}
		}
	}

	free(line);

	return model;
}

inline void ObjLoader::loadMaterial(const char* fileName, Model* model)
{
	if (model->m_materials == NULL)
		model->m_materials = new DynArray<Material>(4);


	FILE *f = 0;
	fopen_s(&f, fileName, "r");
	if (f == NULL)
		throw "Material file not found";

	char lineb[1 << 13];
	char* line;

	Material mat;
	bool readFirstMaterial = true;

	while (fgets(lineb, 1 << 13, f) != NULL)
	{
		line = lineb;
		while (*line == ' ' || *line == '\t') line++;
		if (strncmp(line, "newmtl", 6) == 0)
		{
			if (readFirstMaterial)
			{
				readFirstMaterial = false;
				sscanf_s(&line[7], "%[^\n]", mat.m_name, 256);
				continue;
			}

			int prev = containsSameMaterial(model, &mat);
			if (prev == -1)
			{
				model->m_materials->push_back(mat);
				Material *ptr = &(model->m_materials->at(model->m_materials->size() - 1));
				m_materialMapping->insert(std::pair<const char *, int>(ptr->m_name, model->m_materials->size() - 1));
			}

			mat = Material();
			sscanf_s(&line[7], "%[^\n]", mat.m_name, 256);
		}
		else if (strncmp(line, "Ka", 2) == 0)
		{
			sscanf_s(line, "Ka %f %f %f", &(mat.m_ambient[0]), &(mat.m_ambient[1]), &(mat.m_ambient[2]));
		}
		else if (strncmp(line, "Kd", 2) == 0)
		{
			sscanf_s(line, "Kd %f %f %f", &(mat.m_diffuse[0]), &(mat.m_diffuse[1]), &(mat.m_diffuse[2]));
		}
		else if (strncmp(line, "Ks", 2) == 0)
		{
			sscanf_s(line, "Ks %f %f %f", &(mat.m_specular[0]), &(mat.m_specular[1]), &(mat.m_specular[2]));
		}
		else if (strncmp(line, "d", 1) == 0)
		{
			sscanf_s(line, "d %f", &(mat.m_transparency));
		}
		else if (strncmp(line, "Tr", 2) == 0)
		{
			sscanf_s(line, "Tr %f", &(mat.m_transparency));
		}
		else if (strncmp(line, "Ns", 2) == 0)
		{
			sscanf_s(line, "Ns %f", &(mat.m_shininess));
		}
		else if (strncmp(line, "map_Kd", 6) == 0)
		{
			//sscanf(line, "map_Kd %s", mat.diffuseTexture); //toto neni OK, pokud v nazvu souboru jsou mezery apod.
			sscanf_s(line, "map_Kd %[^\n]", mat.m_diffuseTexture, sizeof(mat.m_diffuseTexture));
			//if (strchr(mat.diffuseTexture, ':') == NULL)  //soubor neni zadan plnou cestou
			//{
			//	char buf[256];
			//	strcpy_s(buf, 256, fileName);
			//	char* ptr = strrchr(buf, '/');
			//	unsigned int restLength = (unsigned int )(ptr - buf + 1);
			//	strcpy_s(ptr+1, 256 - restLength, &mat.diffuseTexture[0]);
			//	buf[strlen(buf)] = 0;
			//	strcpy_s(mat.diffuseTexture, 256, buf);
			//}
		}
		else if (strncmp(line, "illum", 5) == 0)
		{
			sscanf_s(line, "illum %d", &mat.m_illumination);
		}
	}

	//Add last loaded material
	int prev = containsSameMaterial(model, &mat);
	if (prev == -1)
	{
		model->m_materials->push_back(mat);
		Material *ptr = &(model->m_materials->at(model->m_materials->size() - 1));
		m_materialMapping->insert(std::pair<const char *, int>(ptr->m_name, model->m_materials->size() - 1));
	}

	fclose(f);
}

#endif