////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	sceneData.h
//
// summary:	Declaration and implementation of SceneData class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __SCENEDATA_H_
#define __SCENEDATA_H_

#include <dynArray.h>

#include "camera.h"
#include "mouse.h"
#include "light.h"
#include "shader.h"
#include "shaderProgram.h"
#include "model.h"
#include "font.h"
#include "material.h"
#include "entity.h"
#include "vao.h"
#include "fbo.h"
#include "ubo.h"
#include "pbo.h"
#include "boundingBox.h"

//#pragma region FORWARD DECLARATIONS
//class Camera;
//class Mouse;
//struct Light;
//struct Material;
//class Shader;
//class ShaderProgram;
//class Model;
//class Font;
//class Entity;
//class VAO;
//class FBO;
//class UBO;
//class Renderer;
//#pragma endregion

class SceneData
{
private:
	static SceneData *SINGLETON_INSTANCE;
	static SingletonDestroyer<SceneData> destroyer;

public:
	//Shaders
	DynArray<ShaderProgram*> shaderPrograms;
	DynArray<Shader*> vertexShaders;
	DynArray<Shader*> fragmentShaders;
	DynArray<Shader*> geometryShaders;
	DynArray<Shader*> tesselationControlShaders;
	DynArray<Shader*> tesselationEvaluationShaders;

	DynArray<Model*> models;
	
	DynArray<Font*> fonts;
	
	DynArray<Material*> materials;
	DynArray<Light*> lights;

	//openGL textures
	DynArray<GLuint> textures;

	//Scene Entities
	DynArray<Entity*> sceneEntities;
	DynArray<Entity*> infoEntities;
	DynArray<Entity*> idleEntities;
	DynArray<Entity*> nonInitializedEntities;

	//Cameras
	DynArray<Camera*> cameras;

	//VAOs
	DynArray<VAO*> vaos;

	//FBOs
	DynArray<FBO*> fbos;

	//UBOs
	DynArray<UBO*> ubos;

	//UBOs
	DynArray<PBO*> pbos;

	//Mouse
	Mouse*	mouse;

private:

	SceneData() :
		shaderPrograms(DynArray<ShaderProgram*>(16)),
		vertexShaders(DynArray<Shader*>(16)),
		fragmentShaders(DynArray<Shader*>(16)),
		geometryShaders(DynArray<Shader*>(16)),
		tesselationControlShaders(DynArray<Shader*>(16)),
		tesselationEvaluationShaders(DynArray<Shader*>(16)),
		
		models(DynArray<Model*>(16)),
		
		fonts(DynArray<Font*>(4)),

		materials(DynArray<Material*>(16)),

		lights(DynArray<Light*>(4)),

		textures(DynArray<GLuint>(32)),

		sceneEntities(DynArray<Entity*>(16)),
		infoEntities(DynArray<Entity*>(4)),
		idleEntities(DynArray<Entity*>(4)),
		nonInitializedEntities(DynArray<Entity*>(4)),

		cameras(DynArray<Camera*>(2)),

		vaos(DynArray<VAO*>(4)),
		
		fbos(DynArray<FBO*>(4)),

		ubos(DynArray<UBO*>(4)),

		pbos(DynArray<PBO*>(4)),

		mouse(nullptr)
	{

	}

public:
	~SceneData()
	{
		
		for (unsigned int i = 0; i < sceneEntities.size(); i++)
			SAFE_DELETE(sceneEntities[i]);

		for (unsigned int i=0; i<infoEntities.size(); i++)
			SAFE_DELETE(infoEntities[i]);

		for (unsigned int i=0; i<shaderPrograms.size(); i++)
			SAFE_DELETE(shaderPrograms[i]);

		for (unsigned int i=0; i<vertexShaders.size(); i++)
			SAFE_DELETE(vertexShaders[i]);

		for (unsigned int i=0; i<fragmentShaders.size(); i++)
			SAFE_DELETE(fragmentShaders[i]);

		for (unsigned int i=0; i<geometryShaders.size(); i++)
			SAFE_DELETE(geometryShaders[i]);

		for (unsigned int i=0; i<tesselationControlShaders.size(); i++)
			SAFE_DELETE(tesselationControlShaders[i]);

		for (unsigned int i=0; i<tesselationEvaluationShaders.size(); i++)
			SAFE_DELETE(tesselationEvaluationShaders[i]);

		for (unsigned int i=0; i<models.size(); i++)
			SAFE_DELETE(models[i]);

		for (unsigned int i=0; i<fonts.size(); i++)
			SAFE_DELETE(fonts[i]);

		for (unsigned int i=0; i<materials.size(); i++)
			SAFE_DELETE(materials[i]);

		for (unsigned int i = 0; i<lights.size(); i++)
			SAFE_DELETE(lights[i]);

		for (unsigned int i = 0; i<cameras.size(); i++)
			SAFE_DELETE(cameras[i]);

		if (textures.size() > 0)
		{
			glDeleteTextures(textures.size(), textures.front());
		}

		for (unsigned int i = 0; i < vaos.size(); i++)
		{
			SAFE_DELETE(vaos[i]);
		}

		for (unsigned int i = 0; i < fbos.size(); i++)
		{
			SAFE_DELETE(fbos[i]);
		}
		
		for (unsigned int i = 0; i < ubos.size(); i++)
		{
			SAFE_DELETE(ubos[i]);
		}

		for (unsigned int i = 0; i < pbos.size(); i++)
		{
			SAFE_DELETE(pbos[i]);
		}

		SAFE_DELETE(mouse);
	}

	static SceneData* GetInstance()
	{
		if (SINGLETON_INSTANCE == NULL)
		{
			SINGLETON_INSTANCE = new SceneData();
			destroyer.SetDoomed(SINGLETON_INSTANCE);
		}
		return SINGLETON_INSTANCE;
	}
};

#endif