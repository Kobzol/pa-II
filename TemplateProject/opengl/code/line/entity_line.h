#pragma once

#include "stdafx.h"
#include "entity_VAO.h"

#include <code/uniform.h>

class LineEntity : public Entity_VAO
{
public:
	LineEntity(VAO* vao = nullptr) : Entity_VAO(vao) { }
	~LineEntity(void) {}

	void draw(const unsigned int eid = 0);

	glm::vec4 color;
};

inline void LineEntity::draw(const unsigned int eid)
{
	if (!m_isInitialized) return;

	SceneSetting* ss = SceneSetting::GetInstance();
	Uniform<glm::mat4>::bind("MMatrix", ss->m_activeShader->m_programObject, this->m_modelMatrix);
	Uniform<glm::vec4>::bind("Color", ss->m_activeShader->m_programObject, this->color);

	glDisable(GL_CULL_FACE);

	glBindVertexArray(m_vao->m_object);
	glDrawArrays(GL_LINES, 0, 2);
	glBindVertexArray(0);

	glEnable(GL_CULL_FACE);
}
