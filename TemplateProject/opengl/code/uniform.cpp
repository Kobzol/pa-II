#include "uniform.h"

template<>
void Uniform<float>::bindValue(GLuint uniform, const float& value)
{
	glUniform1f(uniform, value);
}

template<>
void Uniform<int>::bindValue(GLuint uniform, const int& value)
{
	glUniform1i(uniform, value);
}

template<>
void Uniform<glm::vec4>::bindValue(GLuint uniform, const glm::vec4& value)
{
	glUniform4fv(uniform, 1, glm::value_ptr(value));
}

template<>
void Uniform<glm::mat4>::bindValue(GLuint uniform, const glm::mat4& value)
{
	glUniformMatrix4fv(uniform, 1, GL_FALSE, glm::value_ptr(value));
}

template<>
void Uniform<glm::mat3>::bindValue(GLuint uniform, const glm::mat3& value)
{
	glUniformMatrix3fv(uniform, 1, GL_FALSE, glm::value_ptr(value));
}
