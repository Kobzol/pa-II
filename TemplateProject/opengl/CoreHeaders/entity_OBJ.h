////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	entity_OBJ.h
//
// summary:	Declaration and implementation of Entity_OBJ class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
/////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __ENTITY_OBJ_H_
#define __ENTITY_OBJ_H_

#include "stdafx.h"

#include <glm.hpp>

#include "entity_VAO.h"
#include "model.h"
#include <CoreHeaders/camera.h>


class Entity_OBJ : public Entity_VAO
{
protected:
	Model *m_model;						//Pointer only. The data is stored in sceneData instance

public :
	Entity_OBJ(Model* model, VAO* vao) : Entity_VAO(vao), m_model(model) {}
	virtual ~Entity_OBJ(void) {}

	virtual void init();
	virtual void draw(const unsigned int eid = 0);
};

inline void Entity_OBJ::init()
{
	if ((m_model) && (m_vao))
	{
		m_isInitialized = true;
	}
}

inline void Entity_OBJ::draw(const unsigned int eid)
{
	if (!m_isInitialized) return;
	
	//Set only juch that uniforms, whose value is related to entity !!!
	SceneSetting *ss = SceneSetting::GetInstance();

	glm::mat4 tmp = ss->m_activeCamera->getVM() * m_modelMatrix;
	int uniform = glGetUniformLocation(ss->m_activeShader->m_programObject, "MMatrix");
	glUniformMatrix4fv(uniform, 1, GL_FALSE, (float*)&m_modelMatrix[0]);
	uniform = glGetUniformLocation(ss->m_activeShader->m_programObject, "MVMatrix");
	glUniformMatrix4fv(uniform, 1, GL_FALSE, (float*)&tmp[0]);
	uniform = glGetUniformLocation(ss->m_activeShader->m_programObject, "NormalMatrix");
	glUniformMatrix3fv(uniform, 1, GL_FALSE, (float*)&(glm::inverseTranspose(glm::mat3(tmp)))[0]);

	glBindVertexArray(m_vao->m_object);

	for (unsigned int i = 0; i<m_vao->m_eai->size(); i++)
	{
		//glActiveTexture(GL_TEXTURE0);
		//glBindTexture(GL_TEXTURE_2D, m_model->m_materials->at(m_vao->m_eai->at(i).m_materialID).m_diffuseTextureGL);
		//Material::setShaderUniform(&(m_model->m_materials->at(m_vao->m_eai->at(i).m_materialID)), ss->m_activeShader, "material");

		glDrawArrays(m_vao->m_eai->at(i).m_glGeometry, m_vao->m_eai->at(i).m_startIndex, m_vao->m_eai->at(i).m_noIndices);
	}

	glBindVertexArray(0); 
}

#endif