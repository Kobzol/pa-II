#include "entity_boid.h"

#include <sceneManager.h>

void Entity_Boid::setTransforms(float3 position, float3 direction, Acceleration acceleration)
{
	SceneManager* manager = SceneManager::GetInstance();

	glm::quat rotation = this->m_orientation;
	if (direction.x != 0.0f || direction.y != 0.0f || direction.z != 0.0f)
	{
		glm::quat targetRotation = glm::rotation(glm::vec3(0.0f, 0.0f, 1.0f),
			glm::normalize(glm::vec3(direction.x, direction.y, direction.z))
		);

		rotation = glm::slerp(this->m_orientation, targetRotation, manager->delta * 2.0f);
	}

	glm::vec3 forces[] = {
		glm::vec3(acceleration.goal.x, acceleration.goal.y, acceleration.goal.z),
		glm::vec3(acceleration.cohesion.x, acceleration.cohesion.y, acceleration.cohesion.z),
		glm::vec3(acceleration.separation.x, acceleration.separation.y, acceleration.separation.z),
		glm::vec3(acceleration.alignment.x, acceleration.alignment.y, acceleration.alignment.z)
	};

	float forceSum = 0.001f;
	for (int i = 0; i < 4; i++)
	{
		forceSum += glm::length(forces[i]);
	}

	for (int i = 0; i < 4; i++)
	{
		float scale = glm::length(forces[i]) / forceSum;

		glm::mat4 model = glm::mat4();

		glm::quat forceRotation = glm::rotation(glm::vec3(0.0f, 0.0f, 1.0f),
			glm::normalize(glm::vec3(forces[i].x, forces[i].y, forces[i].z))
		);

		model = glm::translate(model, glm::vec3(position.x, position.y, position.z));
		model *= glm::mat4_cast(forceRotation);
		model = glm::scale(model, glm::vec3(0.2f + scale));

		this->lineEntities[i]->m_modelMatrix = model;
	}

	glm::mat4 model = glm::mat4();
	model = glm::translate(model, glm::vec3(position.x, position.y, position.z));
	model *= glm::mat4_cast(rotation);
	model = glm::scale(model, glm::vec3(0.1f, 0.1f, 0.1f));

	this->m_modelMatrix = model;
	this->m_orientation = rotation;

	glm::quat viewAngleRotation = glm::rotation(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

	model = glm::mat4();
	model = glm::translate(model, glm::vec3(position.x, position.y, position.z));
	model *= glm::mat4_cast(rotation * viewAngleRotation);
	model = glm::scale(model, glm::vec3(0.4f, 0.4f, 0.4f));
	this->viewAngleEntity->m_modelMatrix = model;
}
