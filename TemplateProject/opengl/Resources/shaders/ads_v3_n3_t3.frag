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

	vec4 ambient = material.ambient * light.ambient;
	vec4 diffuse = material.diffuse * light.diffuse;
	vec4 specular = vec4(0.0f);

	float lightAngle = dot(ecLightDirNormalized, ecNormalNormalized);

	// light side
	if (lightAngle > 0.0f)
	{
		diffuse *= lightAngle;
		// specular
		vec3 halfVector = normalize(ecViewDirNormalized + ecLightDirNormalized);
		float viewAngle = dot(halfVector, ecNormalNormalized);
		specular = (material.specular * light.specular) * pow(viewAngle, light.spotExponent);
	}
	else diffuse = vec4(0.0f);

	FragColor = ambient + diffuse + specular;
}
