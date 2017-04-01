////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	entity_XYGrid.h
//
// summary:	Declares the entity xy grid class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __ENTITY_XYGRID_H_
#define __ENTITY_XYGRID_H_

#include "stdafx.h"

#include <glm.hpp>

#include "material.h"
#include "entity_VAO.h"
#include "vao_GridXY.h"

class Entity_GridXY : public Entity_VAO
{
private:
	int m_noPositiveCells;

public:
	Entity_GridXY(VAO* vao = nullptr) : Entity_VAO(vao), m_noPositiveCells(0)
	{ 
		m_castShadows = false;
		if (vao)
		{
			VAO_GridXY *p = dynamic_cast<VAO_GridXY*>(vao);
			m_noPositiveCells = p->m_noPositiveCells;
		}
	}
	~Entity_GridXY(void){}

	void draw(const unsigned int eid = 0);
};

inline void Entity_GridXY::draw(const unsigned int eid)
{
	if (!m_isInitialized) return;
	
	glBindVertexArray(m_vao->m_object);
	glDrawArrays(GL_LINES, 0, 4*(2*m_noPositiveCells+1));
	glBindVertexArray(0); 
}

#endif