#version 420 core

layout(location = 0) in vec3 VertexPosition;
layout(location = 1) in vec2 VertexTex;

uniform mat4 VMatrix;
uniform mat4 PMatrix;
uniform mat4 MMatrix;

out vec2 texCoords;
out vec2 centerVector;

void main()
{
	texCoords = VertexTex;
	centerVector = VertexPosition.xy;

	gl_Position = PMatrix * VMatrix * MMatrix * vec4(VertexPosition, 1.0);
}
