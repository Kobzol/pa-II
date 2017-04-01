#version 400 core //compatibility

#include "structures.header"

layout(location = 0) out vec4 FragColor;

uniform LightInfo light;
uniform MaterialInfo material;

//The prefix ec means Eye Coordinates in the Eye Coordinate System
in vec4 ecPosition;			
in vec3 ecLightDir;
in vec3 ecNormal;
in vec3 ecViewDir;

void main()
{
	vec3 ecLightDirNormalized = normalize(ecLightDir);
	vec3 ecNormalNormalized = normalize(ecNormal);
	vec3 ecViewDirNormalized = normalize(ecViewDir);

	vec4 to = vec4(0.0f, 0.0f, 1.0f, 1.0f);
	vec4 from = vec4(1.0f, 0.0f, 0.0f, 1.0f);

	float val = dot(ecViewDirNormalized, ecNormalNormalized);

	vec3 halfVector = normalize(ecViewDirNormalized + ecLightDirNormalized);
	float viewAngle = dot(halfVector, ecNormalNormalized);

	vec4 specular = vec4(0.0f);
	if (viewAngle > 0.0f)
	{
		specular = (material.specular * light.specular) * pow(viewAngle, light.spotExponent);
	}

	if (abs(val) < 0.15f)
	{
		FragColor = vec4(0.0f);
	}
	else FragColor = mix(from, to, val) + specular;
}
