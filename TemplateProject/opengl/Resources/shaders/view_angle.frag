#version 420 core

layout(location = 0) out vec4 FragColor;

in vec2 texCoords;
in vec2 centerVector;

uniform float ViewAngle;

void main()
{
	vec2 fromCenter = normalize(centerVector);
	float angle = abs(atan(fromCenter.y, fromCenter.x));
	
	float alpha = (ViewAngle - angle) / ViewAngle; 

	if (angle <= ViewAngle && length(centerVector) <= 0.5f)
	{
		FragColor = vec4(0.0f, 1.0f, 0.0f, alpha);
	}
	else discard;
}
