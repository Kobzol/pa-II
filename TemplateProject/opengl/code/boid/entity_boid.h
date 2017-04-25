#pragma once

#include <random>
#include <ctime>

#include "../CoreHeaders/entity_OBJ.h"
#include "../quad/vao_quad.h"
#include "../quad/entity_view_angle.h"
#include "../line/vao_line.h"
#include "../line/entity_line.h"
#include "../../../boids/boids.h"
#include <sceneManager.h>

static std::default_random_engine engine((unsigned int) time(nullptr));
static std::normal_distribution<float> distribution(1.5f, 0.5f);

class Entity_Boid : public Entity_OBJ
{
public:
	Entity_Boid(Model* model, VAO* vao, VAO* quadVAO, ShaderProgram* quadShader,
		VAO* lineVAO, ShaderProgram* lineShader) : Entity_OBJ(model, vao), quadShader(quadShader), lineShader(lineShader)
	{
		glm::vec4 colors[] = {
			glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
			glm::vec4(0.0f, 1.0f, 1.0f, 1.0f),
			glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
			glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)
		};
		for (int i = 0; i < 4; i++)
		{
			LineEntity* entity = new LineEntity(lineVAO);
			entity->init();
			entity->color = colors[i];
			this->lineEntities.push_back(entity);
		}
		this->viewAngleEntity = new ViewAngleEntity(quadVAO);
		this->viewAngleEntity->init();
	}
	~Entity_Boid(void) {}

	void draw(const unsigned int eid = 0);

	void setViewAngle(float degrees)
	{
		this->viewAngleEntity->setViewAngle(degrees);
	}
	void setDrawHelper(bool value)
	{
		this->drawHelper = value;
	}
	void setTransforms(float3 position, float3 direction, Acceleration acceleration);
private:
	void drawViewAngle();
	void drawLineAngles();

	void updateWings();
	void switchWingsDirection();

	ShaderProgram* quadShader;
	ShaderProgram* lineShader;

	bool drawHelper = false;

	float wingsSpeed = 1.5f;
	float wingsState = 0.0f;
	bool wingsUp = true;

	ViewAngleEntity* viewAngleEntity;
	std::vector<LineEntity*> lineEntities;
};

inline void Entity_Boid::updateWings()
{
	float delta = SceneManager::GetInstance()->delta * this->wingsSpeed;
	if (this->wingsUp)
	{
		this->wingsState += delta;
		if (this->wingsState >= 1.0f)
		{
			this->switchWingsDirection();
		}
	}
	else
	{
		this->wingsState -= delta;
		if (this->wingsState <= 0.0f)
		{
			this->switchWingsDirection();
		}
	}

	this->wingsState = glm::clamp(this->wingsState, 0.0f, 1.0f);
}
inline void Entity_Boid::switchWingsDirection()
{
	this->wingsUp = !this->wingsUp;
	this->wingsSpeed = glm::clamp(distribution(engine), 0.5f, 2.0f);
}

inline void Entity_Boid::draw(const unsigned int eid)
{
	if (!m_isInitialized) return;

	this->updateWings();

	SceneSetting* ss = SceneSetting::GetInstance();
	Uniform<float>::bind("VertexMix", ss->m_activeShader->m_programObject, this->wingsState);

	Entity_OBJ::draw(eid);

	if (this->drawHelper)
	{
		this->drawLineAngles();
		this->drawViewAngle();
	}
}

inline void Entity_Boid::drawViewAngle()
{
	SceneSetting* ss = SceneSetting::GetInstance();
	ss->m_activeShader = this->quadShader;
	ss->m_activeShader->enable();

	this->viewAngleEntity->draw();
}
inline void Entity_Boid::drawLineAngles()
{
	SceneSetting* ss = SceneSetting::GetInstance();
	ss->m_activeShader = this->lineShader;
	ss->m_activeShader->enable();

	for (int i = 0; i < this->lineEntities.size(); i++)
	{
		this->lineEntities[i]->draw();
	}
}
