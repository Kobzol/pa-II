#version 420 core

in vec2 texCoords;

uniform sampler2D texSampler;

layout(location = 0) out vec4 FragColor;

void main()
{
	FragColor = texture(texSampler, texCoords);
}
