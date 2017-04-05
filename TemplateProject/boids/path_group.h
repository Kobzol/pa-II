#pragma once

#include <vector>
#include <glm/glm.hpp>

class PathGroup
{
public:
	PathGroup(float distanceLimit, std::vector<glm::vec3> positions) : distanceLimit(distanceLimit), positions(positions)
	{

	}

	void update(const glm::vec3& position)
	{
		glm::vec3 currentPos = this->getCurrentTarget();
		if (glm::distance(currentPos, position) < this->distanceLimit)
		{
			this->moveTarget();
		}
	}

	glm::vec3 getCurrentTarget() const
	{
		return this->positions[this->currentPosition];
	}

private:
	int currentPosition = 0;
	float distanceLimit;

	void moveTarget()
	{
		this->currentPosition = (this->currentPosition + 1) % this->positions.size();
	}

	std::vector<glm::vec3> positions;
};
