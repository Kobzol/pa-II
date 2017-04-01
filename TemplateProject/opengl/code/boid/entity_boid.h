#pragma once

#include "../CoreHeaders/entity_OBJ.h"
#include "../quad/vao_quad.h"

class Entity_Boid : public Entity_OBJ
{
public:
	Entity_Boid(Model* model, VAO* vao, VAO* quadVAO, ShaderProgram* quadShader) : Entity_OBJ(model, vao)
	{
		this->quadVAO = quadVAO;
		this->quadShader = quadShader;
	}
	~Entity_Boid(void) {}

	void draw(const unsigned int eid = 0);

	void setViewAngle(float degrees)
	{
		this->viewAngle = glm::radians(fmod(degrees, 180.1f));
	}

	VAO* quadVAO;
	ShaderProgram* quadShader;
	glm::mat4 viewAngleModelMatrix;
private:
	void drawViewAngle();

	float viewAngle;
};

inline void Entity_Boid::draw(const unsigned int eid)
{
	if (!m_isInitialized) return;

	Entity_OBJ::draw(eid);

	this->drawViewAngle();
}

inline void Entity_Boid::drawViewAngle()
{
	SceneSetting* ss = SceneSetting::GetInstance();
	ss->m_activeShader = this->quadShader;
	ss->m_activeShader->enable();

	Uniform<glm::mat4>::bind("MMatrix", ss->m_activeShader->m_programObject, this->viewAngleModelMatrix);
	Uniform<glm::mat4>::bind("PMatrix", ss->m_activeShader->m_programObject, ss->m_activeCamera->getProjectionMatrix());
	Uniform<glm::mat4>::bind("VMatrix", ss->m_activeShader->m_programObject, ss->m_activeCamera->getViewMatrix());

	Uniform<float>::bind("ViewAngle", ss->m_activeShader->m_programObject, this->viewAngle);

	glDisable(GL_CULL_FACE);

	glBindVertexArray(this->quadVAO->m_object);
	glDrawArrays(GL_QUADS, 0, 4);
	glBindVertexArray(0);

	glEnable(GL_CULL_FACE);
}
