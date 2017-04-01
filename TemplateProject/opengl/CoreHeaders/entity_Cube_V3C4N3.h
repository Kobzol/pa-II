#pragma once

#include "stdafx.h"
#include "entity_VAO.h"

class Entity_Cube_V3C4N3 : public Entity_VAO
{
public:
	Entity_Cube_V3C4N3(VAO* vao = nullptr) : Entity_VAO(vao) {}
	~Entity_Cube_V3C4N3(void) {}

	void draw(const unsigned int eid = 0);
};

inline void Entity_Cube_V3C4N3::draw(const unsigned int eid)
{
	if (!m_isInitialized) return;

	glBindVertexArray(m_vao->m_object);
	glDrawArrays(GL_QUADS, 0, 24);
	glBindVertexArray(0);
}
