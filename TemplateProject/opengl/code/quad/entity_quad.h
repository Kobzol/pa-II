#pragma once

#include "stdafx.h"
#include "entity_VAO.h"

#include <code/uniform.h>

class QuadEntity : public Entity_VAO
{
public:
	QuadEntity(VAO* vao = nullptr) : Entity_VAO(vao)
	{
		this->setViewAngle(45.0f);
	}
	~QuadEntity(void) {}

	void draw(const unsigned int eid = 0);

	void setViewAngle(float degrees)
	{
		this->viewAngle = glm::radians(fmod(degrees, 180.0f));
	}

private:
	float viewAngle;
};

inline void QuadEntity::draw(const unsigned int eid)
{
	if (!m_isInitialized) return;

	SceneSetting* ss = SceneSetting::GetInstance();
	Uniform<glm::mat4>::bind("MMatrix", ss->m_activeShader->m_programObject, this->m_modelMatrix);
	Uniform<glm::mat4>::bind("PMatrix", ss->m_activeShader->m_programObject, ss->m_activeCamera->getProjectionMatrix());
	Uniform<glm::mat4>::bind("VMatrix", ss->m_activeShader->m_programObject, ss->m_activeCamera->getViewMatrix());

	Uniform<float>::bind("ViewAngle", ss->m_activeShader->m_programObject, this->viewAngle);

	glDisable(GL_CULL_FACE);

	glBindVertexArray(m_vao->m_object);
	glDrawArrays(GL_QUADS, 0, 4);
	glBindVertexArray(0);

	glEnable(GL_CULL_FACE);
}
#pragma once
