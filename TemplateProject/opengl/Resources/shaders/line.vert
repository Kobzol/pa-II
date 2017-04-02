#version 420 core

layout(location = 0) in vec3 VertexPosition;

uniform mat4 VMatrix;
uniform mat4 PMatrix;
uniform mat4 MMatrix;

void main()
{
	gl_Position = PMatrix * VMatrix * MMatrix * vec4(VertexPosition, 1.0);
}
