////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	entity_SceneOrigin.h
//
// summary: Declaration and implementation of Entity_SceneOrigin class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __ENTITY_SCENEORIGIN_H_
#define __ENTITY_SCENEORIGIN_H_

#include "stdafx.h"
#include "entity_VAO.h"

class Entity_SceneOrigin : public Entity_VAO
{
public:
	Entity_SceneOrigin(VAO* vao = nullptr) : Entity_VAO(vao) { m_castShadows = false; }
	~Entity_SceneOrigin(void){}

	void draw(const unsigned int eid = 0);
};

inline void Entity_SceneOrigin::draw(const unsigned int eid)
{
	if (!m_isInitialized) return;
	
	glBindVertexArray(m_vao->m_object);
	glDrawArrays(GL_LINES, 0, 6);
	glBindVertexArray(0); 
}

#endif