#pragma once

#include "stdafx.h"
#include "entity_VAO.h"

class TetrahedraEntity : public Entity_VAO
{
public:
	TetrahedraEntity(VAO* vao = nullptr) : Entity_VAO(vao) {}
	~TetrahedraEntity(void) {}

	void draw(const unsigned int eid = 0);
};

inline void TetrahedraEntity::draw(const unsigned int eid)
{
	if (!m_isInitialized) return;

	glBindVertexArray(m_vao->m_object);
	glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}
