#version 400 core //compatibility

#include "structures.header"

layout(location = 0) in vec3 VertexPosition;
layout(location = 1) in vec3 VertexNormal;
layout(location = 2) in vec3 VertexTex;

uniform mat4 PMatrix;		//Camera projection matrix
uniform mat4 VMatrix;		//Camera view matrix
uniform mat4 MVMatrix;		//VMatrix * Model matrix
uniform mat3 NormalMatrix;	//MVMatrix ... -> converted into normal matrix (inverse transpose operation)

uniform LightInfo light;

//The prefix ec means Eye Coordinates in the Eye Coordinate System
out vec4 ecPosition;			
out vec3 ecLightDir;
out vec3 ecNormal;
out vec3 ecViewDir;

void main()
{
	ecPosition = MVMatrix * vec4(VertexPosition, 1.0f);
	ecLightDir = vec3((VMatrix * light.position) - ecPosition);
	ecNormal = NormalMatrix * VertexNormal;
	ecViewDir = -vec3(ecPosition);	// kamera je v (0, 0, 0)

	gl_Position = PMatrix * ecPosition;
}
