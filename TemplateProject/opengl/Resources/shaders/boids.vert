#version 420 core

layout(location = 0) in vec3 VertexPosition;
layout(location = 1) in vec3 VertexNormal;
layout(location = 2) in vec3 VertexTex;
layout(location = 3) in vec3 VertexPositionModified;

uniform mat4 VMatrix;
uniform mat4 PMatrix;
uniform mat4 MMatrix;

uniform float VertexMix;

void main()
{
	gl_Position = PMatrix * VMatrix * MMatrix * vec4(mix(VertexPosition, VertexPositionModified, VertexMix), 1.0);
}
