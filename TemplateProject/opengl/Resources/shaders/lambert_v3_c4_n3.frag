#version 400 core //compatibility

#include "structures.header"

layout(location = 0) out vec4 FragColor;

uniform LightInfo light;

//The prefix ec means Eye Coordinates in the Eye Coordinate System
in vec4 ecPosition;			
in vec3 ecLightDir;
in vec3 ecNormal;
in vec3 ecViewDir;

void main()
{
	vec3 ecLightDirNormalized = normalize(ecLightDir);
	vec3 ecNormalNormalized = normalize(ecNormal);
	vec4 color = vec4(1.0f, 0.0f, 0.0f, 1.0f);

	float nLightAngle = dot(ecLightDirNormalized, ecNormalNormalized);

	if (nLightAngle > 0)
	{
		color = vec4(0.0f, 1.0f, 0.0f, 1.0f);
	}

	FragColor = color;
}
