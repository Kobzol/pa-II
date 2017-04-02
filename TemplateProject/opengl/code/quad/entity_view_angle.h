#pragma once

#include "stdafx.h"
#include "entity_VAO.h"

#include <code/uniform.h>

class ViewAngleEntity : public Entity_VAO
{
public:
	ViewAngleEntity(VAO* vao = nullptr) : Entity_VAO(vao)
	{
		this->setViewAngle(45.0f);
	}
	~ViewAngleEntity(void) {}

	void draw(const unsigned int eid = 0);

	void setViewAngle(float degrees)
	{
		this->viewAngle = glm::radians(fmod(degrees, 180.0f));
	}

private:
	float viewAngle;
};

inline void ViewAngleEntity::draw(const unsigned int eid)
{
	if (!m_isInitialized) return;

	SceneSetting* ss = SceneSetting::GetInstance();
	Uniform<glm::mat4>::bind("MMatrix", ss->m_activeShader->m_programObject, this->m_modelMatrix);
	Uniform<float>::bind("ViewAngle", ss->m_activeShader->m_programObject, this->viewAngle);

	glDisable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glBindVertexArray(m_vao->m_object);
	glDrawArrays(GL_QUADS, 0, 4);
	glBindVertexArray(0);

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
}
#pragma once
