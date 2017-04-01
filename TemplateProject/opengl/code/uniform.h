#pragma once

#include <iostream>

#include <glew.h>
#include <glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <sceneSetting.h>

template <typename T>
class Uniform
{
public:
	static void bind(const std::string& name, SceneSetting* setting, const T& value)
	{
		Uniform<T>::bind(name, setting->m_activeShader->m_programObject, value);
	}
	static void bind(const std::string& name, GLuint program, const T& value)
	{
		GLint uniform;
		if ((uniform = glGetUniformLocation(program, name.c_str())) >= 0)
		{
			Uniform<T>::bindValue(uniform, value);
		}
		else std::cerr << "No uniform named " << name << " found" << std::endl;
	}

	static void bindValue(GLuint uniform, const T& value);
};
