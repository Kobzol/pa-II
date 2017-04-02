#pragma once

#include <sceneInitializer.h>

#include <Entity_OBJ.h>
#include "../code/boid/entity_boid.h"

class DemoBoids : public SceneInitializer
{
private:
	int boidCount;

	void initShaders();
	void initModels();
	void initVAOs();
	void initMaterials();
	void initInfoEntities();
	void initSceneEntities();

public:
	DemoBoids(SceneData *sdPtr, int boidCount) : SceneInitializer(sdPtr), boidCount(boidCount)
	{
		
	}

	void render();

	std::vector<Entity_Boid*> boids;
	std::vector<Entity_OBJ*> modelObjects;
};
