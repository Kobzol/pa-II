////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	sceneInitializer.h
//
// summary:	Declaration and implementation of SceneInitializer class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __SCENEINITIALIZER_H_
#define __SCENEINITIALIZER_H_

#include <dynArray.h>
#include <math.h>
#include <FreeImage.h>

#include "imageManager.h"
#include "objLoader.h"

#include "sceneData.h"

#include "guiManager.h"

#include "camera.h"
#include "light.h"
#include "mouse.h"
#include "entity.h"
#include "shaderProgram.h"
#include "model.h"
#include "font.h"

#include "helper.h"

#define DEFAULT_RESOURCE_PATH "opengl/Resources/"

class SceneInitializer
{
protected:
	SceneData *m_sceneData;
	char m_resourcePath[1024];
	char m_resourceFile[1024];

protected:
	virtual void initCameras();

	virtual void initShaders() {}
	virtual void initTextures() {}
	virtual void initModels() {}
	virtual void initFonts() {}
	virtual void initMaterials() {}
	virtual void initLights();

	virtual void initInfoEntities() {}
	virtual void initSceneEntities() {}

	virtual void initVAOs() {}
	virtual void initFBOs() {}
	virtual void initUBOs() {}
	virtual void initPBOs() {}

public:
	virtual void init()
	{
		initCameras();

		initShaders();
		initTextures();
		initModels();
		initFonts();
		initMaterials();
		initLights();

		initVAOs(); 
		initFBOs();
		initUBOs();
		initPBOs();

		initInfoEntities();
		initSceneEntities();
	}

	virtual void render()
	{
		for (unsigned int i = 0; i < m_sceneData->infoEntities.size(); i++)
			m_sceneData->infoEntities[i]->draw();

		for (unsigned int i = 0; i < m_sceneData->sceneEntities.size(); i++)
			m_sceneData->sceneEntities[i]->draw(i);
	}

protected:
	SceneInitializer(SceneData *sdPtr) : m_sceneData(sdPtr)
	{
		strcpy_s(m_resourcePath, 1024, DEFAULT_RESOURCE_PATH);
		strcpy_s(m_resourceFile, 1024, "");
	}
	void initShaderProgram(const char* vertexShader, const char* fragmentShader);
	void initShaderProgram(const char* vertexShader, const char* fragmentShader, const char* geometryShader);
	void initShaderProgram(const char* vertexShader, const char* fragmentShader, const char* geometryShader, const char* tesselationControlShader, const char* tesselationEvaluationShader);
	bool checkFramebufferStatus();

	void resetResPath();
	void addResPath(const char *_resourcePath);
	const char* getResFile(const char *resourceFileName);

	void addUnifyMaterials(Model *m);

public:
	virtual ~SceneInitializer() {}

};

inline void SceneInitializer::initCameras()
{
	Camera *c = new Camera();
	m_sceneData->cameras.push_back(c);
}

inline void SceneInitializer::initLights()
{
	Light *l = new Light();
	m_sceneData->lights.push_back(l);
}


inline void SceneInitializer::resetResPath()
{
	strcpy_s(m_resourcePath, 1024, DEFAULT_RESOURCE_PATH);
}

inline void SceneInitializer::addResPath(const char *resourcePath)
{
	strcat_s(m_resourcePath, 1024, resourcePath);
}

inline const char* SceneInitializer::getResFile(const char *resourceFileName)
{
	strcpy_s(m_resourceFile, 1024, m_resourcePath);
	strcat_s(m_resourceFile, 1024, resourceFileName);
	return m_resourceFile;
}

inline void SceneInitializer::initShaderProgram(const char* vertexShader, const char* fragmentShader)
{
	Shader *s;
	ShadersFlag flag = SF_NONE;
	if (vertexShader)
	{
		s = new Shader(GL_VERTEX_SHADER);
		s->openFromFile(getResFile(vertexShader));
		m_sceneData->vertexShaders.push_back(s);
		flag = (ShadersFlag)(flag | SF_VERTEX);
	}

	if (fragmentShader)
	{
		s = new Shader(GL_FRAGMENT_SHADER);
		s->openFromFile(getResFile(fragmentShader));
		m_sceneData->fragmentShaders.push_back(s);	
		flag = (ShadersFlag)(flag | SF_FRAGMENT);
	}

	ShaderProgram *sp = new ShaderProgram(	(flag & SF_VERTEX) ? m_sceneData->vertexShaders.at(m_sceneData->vertexShaders.size()-1) : 0,
											(flag & SF_FRAGMENT) ? m_sceneData->fragmentShaders.at(m_sceneData->fragmentShaders.size()-1) : 0);
	m_sceneData->shaderPrograms.push_back(sp);
}

inline void SceneInitializer::initShaderProgram(const char* vertexShader, const char* fragmentShader, const char* geometryShader)
{
	Shader *s;
	ShadersFlag flag = SF_NONE;
	if (vertexShader)
	{
		s = new Shader(GL_VERTEX_SHADER);
		s->openFromFile(getResFile(vertexShader));
		m_sceneData->vertexShaders.push_back(s);
		flag = (ShadersFlag)(flag | SF_VERTEX);
	}

	if (fragmentShader)
	{
		s = new Shader(GL_FRAGMENT_SHADER);
		s->openFromFile(getResFile(fragmentShader));
		m_sceneData->fragmentShaders.push_back(s);	
		flag = (ShadersFlag)(flag | SF_FRAGMENT);
	}
	
	if (geometryShader)
	{
		s = new Shader(GL_GEOMETRY_SHADER);
		s->openFromFile(getResFile(geometryShader));
		m_sceneData->geometryShaders.push_back(s);	
		flag = (ShadersFlag)(flag | SF_GEOMETRY);
	}

	ShaderProgram *sp = new ShaderProgram(	(flag & SF_VERTEX) ? m_sceneData->vertexShaders.at(m_sceneData->vertexShaders.size()-1) : 0,
											(flag & SF_FRAGMENT) ? m_sceneData->fragmentShaders.at(m_sceneData->fragmentShaders.size()-1) : 0,
											(flag & SF_GEOMETRY) ? m_sceneData->geometryShaders.at(m_sceneData->geometryShaders.size()-1) : 0);
	m_sceneData->shaderPrograms.push_back(sp);
}

inline void SceneInitializer::initShaderProgram(const char* vertexShader, const char* fragmentShader, const char* geometryShader, const char* tesselationControlShader, const char* tesselationEvaluationShader)
{
	Shader *s;
	ShadersFlag flag = SF_NONE;
	if (vertexShader)
	{
		s = new Shader(GL_VERTEX_SHADER);
		s->openFromFile(getResFile(vertexShader));
		m_sceneData->vertexShaders.push_back(s);
		flag = (ShadersFlag)(flag | SF_VERTEX);
	}

	if (fragmentShader)
	{
		s = new Shader(GL_FRAGMENT_SHADER);
		s->openFromFile(getResFile(fragmentShader));
		m_sceneData->fragmentShaders.push_back(s);	
		flag = (ShadersFlag)(flag | SF_FRAGMENT);
	}
	
	if (geometryShader)
	{
		s = new Shader(GL_GEOMETRY_SHADER);
		s->openFromFile(getResFile(geometryShader));
		m_sceneData->geometryShaders.push_back(s);	
		flag = (ShadersFlag)(flag | SF_GEOMETRY);
	}

	if (tesselationControlShader)
	{
		s = new Shader(GL_TESS_CONTROL_SHADER);
		s->openFromFile(getResFile(tesselationControlShader));
		m_sceneData->tesselationControlShaders.push_back(s);	
		flag = (ShadersFlag)(flag | SF_TESSELATION_CONTROL);
	}

	if (tesselationEvaluationShader)
	{
		s = new Shader(GL_TESS_EVALUATION_SHADER);
		s->openFromFile(getResFile(tesselationEvaluationShader));
		m_sceneData->tesselationEvaluationShaders.push_back(s);	
		flag = (ShadersFlag)(flag | SF_TESSELATION_EVALUATION);
	}

	ShaderProgram *sp = new ShaderProgram(	(flag & SF_VERTEX) ? m_sceneData->vertexShaders.at(m_sceneData->vertexShaders.size()-1) : 0,
											(flag & SF_FRAGMENT) ? m_sceneData->fragmentShaders.at(m_sceneData->fragmentShaders.size()-1) : 0,
											(flag & SF_GEOMETRY) ? m_sceneData->geometryShaders.at(m_sceneData->geometryShaders.size()-1) : 0,
											(flag & SF_TESSELATION_CONTROL) ? m_sceneData->tesselationControlShaders.at(m_sceneData->tesselationControlShaders.size()-1) : 0,
											(flag & SF_TESSELATION_EVALUATION) ? m_sceneData->tesselationEvaluationShaders.at(m_sceneData->tesselationEvaluationShaders.size()-1) : 0);
	m_sceneData->shaderPrograms.push_back(sp);
}

inline bool SceneInitializer::checkFramebufferStatus()
{
	// check FBO status
	GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
	switch(status)
	{
	case GL_FRAMEBUFFER_COMPLETE_EXT:
		printf("Framebuffer complete.");
		return true;

	case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
		printf("[ERROR] Framebuffer incomplete: Attachment is NOT complete.");
		return false;

	case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
		printf("[ERROR] Framebuffer incomplete: No image is attached to FBO.");
		return false;

	case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
		printf("[ERROR] Framebuffer incomplete: Attached images have different dimensions.");
		return false;

	case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
		printf("[ERROR] Framebuffer incomplete: Color attached images have different internal formats.");
		return false;

	case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
		printf("[ERROR] Framebuffer incomplete: Draw buffer.");
		return false;

	case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
		printf("[ERROR] Framebuffer incomplete: Read buffer.");
		return false;

	case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
		printf("[ERROR] Unsupported by FBO implementation.");
		return false;

	default:
		printf("[ERROR] Unknow error.");
		return false;
	}
}

inline void SceneInitializer::addUnifyMaterials(Model *m)
{
	Material *mat;

	if (m)
	{
		if (m->m_materials)
		{
			for (unsigned int i = 0; i<m->m_materials->size(); i++)
			{
				bool isUnique = true;
				for (unsigned int j = 0; j < m_sceneData->materials.size(); j++)
				{
					if (m->m_materials->at(i) == *(m_sceneData->materials.at(j)))
					{
						isUnique = false;
					}
				}
				
				if (isUnique)
				{
					mat = new Material(*(m_sceneData->materials.at(i)));
					m_sceneData->materials.push_back(mat);
				}
			}
		}
	}
}
#endif
