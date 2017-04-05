#include "demo_boids.h"

#include "vao_SceneOrigin.h"
#include "vao_GridXY.h"
#include "vao_CubeV3C4.h"

#include "entity_SceneOrigin.h"
#include "entity_GridXY.h"
#include "entity_Cube_Simple.h"
#include "entity_OBJ.h"
#include <sceneManager.h>

#include "../code/uniform.h"
#include "../code/boid/entity_boid.h"
#include "../code/quad/vao_quad.h"
#include "../code/line/vao_line.h"
#include "../code/line/entity_line.h"

#define SHADER_BOID (0)
#define SHADER_PHONG (1)
#define SHADER_VIEW_ANGLE (2)
#define SHADER_LINE (3)

#define MODEL_BOID (0)
#define MODEL_BOID_INTER (1)
#define MODEL_VASE (2)

#define VAO_BOID (0)
#define VAO_VASE (1)
#define VAO_QUAD (2)
#define VAO_LINE (3)

void DemoBoids::initShaders()
{
	addResPath("shaders/");
	initShaderProgram("boids.vert", "boids.frag");
	initShaderProgram("ads_v3_n3_t3.vert", "ads_v3_n3_t3.frag");
	initShaderProgram("quad.vert", "view_angle.frag");
	initShaderProgram("line.vert", "line.frag");
	resetResPath();
}

void DemoBoids::initModels()
{
	ObjLoader objL;
	Model* m;

	addResPath("models/");

	m = objL.loadModel(getResFile("cone/boid_state_1.obj"));
	m_sceneData->models.push_back(m);

	m = objL.loadModel(getResFile("cone/boid_state_2.obj"));
	m_sceneData->models.push_back(m);

	m = objL.loadModel(getResFile("vase/vase.obj"));
	m_sceneData->models.push_back(m);

	resetResPath();
}

void DemoBoids::initVAOs()
{
	VAO* vao = new VAO();
	vao->createFromModelInterpolated(m_sceneData->models[MODEL_BOID], m_sceneData->models[MODEL_BOID_INTER]);
	m_sceneData->vaos.push_back(vao);

	vao = new VAO();
	vao->createFromModel(m_sceneData->models[MODEL_VASE]);
	m_sceneData->vaos.push_back(vao);

	vao = new QuadVAO();
	vao->init();
	m_sceneData->vaos.push_back(vao);

	vao = new LineVAO();
	vao->init();
	m_sceneData->vaos.push_back(vao);
}

void DemoBoids::initMaterials()
{
	Material *m = new Material();

	m->setName("White_opaque");
	m->m_diffuse[0] = 1.0f;
	m->m_diffuse[1] = 1.0f;
	m->m_diffuse[2] = 1.0f;
	m->m_diffuse[3] = 1.0f;
	m->m_transparency = 0.0f;

	m_sceneData->materials.push_back(m);
}


void DemoBoids::initInfoEntities()
{
	
}

void DemoBoids::initSceneEntities()
{
	for (int i = 0; i < this->boidCount; i++)
	{
		Entity_Boid *e1 = new Entity_Boid(m_sceneData->models[MODEL_BOID], m_sceneData->vaos[VAO_BOID],
			m_sceneData->vaos[VAO_QUAD], m_sceneData->shaderPrograms[SHADER_VIEW_ANGLE],
			m_sceneData->vaos[VAO_LINE], m_sceneData->shaderPrograms[SHADER_LINE]);
		e1->setPosition(0.0f, 0.0, 0.0f);
		e1->init();

		m_sceneData->sceneEntities.push_back(e1);
		this->boids.push_back(e1);
	}

	Entity_OBJ *obj = new Entity_OBJ(m_sceneData->models[MODEL_VASE], m_sceneData->vaos[VAO_VASE]);
	obj->setPosition(0.0f, 0.0, 100.0f);
	obj->setOrientation(0.0f, 0.0f, 90.0f);
	obj->m_material = m_sceneData->materials[0];
	obj->init();
	m_sceneData->sceneEntities.push_back(obj);
	this->modelObjects.push_back(obj);
}

void DemoBoids::render()
{
	SceneSetting* ss = SceneSetting::GetInstance();
	for (int i = 0; i < m_sceneData->shaderPrograms.size(); i++)
	{
		ShaderProgram* sp = m_sceneData->shaderPrograms[i];
		sp->enable();
		Uniform<glm::mat4>::bind("PMatrix", sp->m_programObject, ss->m_activeCamera->getProjectionMatrix());
		Uniform<glm::mat4>::bind("VMatrix", sp->m_programObject, ss->m_activeCamera->getViewMatrix());
	}

#pragma region Boids
	ss->m_activeShader = m_sceneData->shaderPrograms[SHADER_BOID];
	ss->m_activeShader->enable();

	static float counter = 0.0f;
	static bool up = true;
	float delta = SceneManager::GetInstance()->delta * 3.0f;
	if (up)
	{
		counter += delta;
		if (counter >= 1.0f) up = false;
	}
	else
	{
		counter -= delta;
		if (counter <= 0.0f) up = true;
	}

	Uniform<float>::bind("VertexMix", ss->m_activeShader->m_programObject, counter);

	for (unsigned int i = 0; i < this->boids.size(); i++)
	{
		ss->m_activeShader = m_sceneData->shaderPrograms[SHADER_BOID];
		ss->m_activeShader->enable();
		this->boids[i]->draw(i);
	}
#pragma endregion
	
#pragma region ModelObjects
	ss->m_activeShader = m_sceneData->shaderPrograms[SHADER_PHONG];
	ss->m_activeShader->enable();

	Light::setShaderUniform(m_sceneData->lights.at(0), ss->m_activeShader, "light");

	for (unsigned int i = 0; i < this->modelObjects.size(); i++)
	{
		Entity* obj = this->modelObjects[i];
		Material::setShaderUniform(obj->m_material, ss->m_activeShader, "material");
		obj->draw(i);
	}
#pragma endregion
}
