////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	entity_VAO.h
//
// summary:	Declaration and implementation of Entity_VAO class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __ENTITY_VBO_H_
#define __ENTITY_VBO_H_

#include "stdafx.h"
#include "entity.h"
#include "vao.h"

class Entity_VAO : public Entity
{
public:
	VAO *m_vao;						//Pointer only. The data is stored in sceneData instance

public :
	Entity_VAO(VAO* vao = nullptr) : Entity(), m_vao(vao) {}

	virtual ~Entity_VAO(void) {}

	virtual void init() 
	{ 
		if (m_vao)
			m_isInitialized = true; 
	}

	virtual void draw(const unsigned int eid = 0) {}
};

#endif