#version 420 core

layout(location = 0) in vec3  VertexPosition;
layout(location = 1) in vec4  VertexColor;

uniform mat4 MVPMatrix;
out vec4 color;

void main()
{
	color = VertexColor;
	gl_Position = MVPMatrix * vec4(VertexPosition, 1.0);
	//gl_Position = PMatrix * MVMatrix * VertexPosition;
}
