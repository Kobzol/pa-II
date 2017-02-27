#pragma once

#include <glew.h>
#include <freeglut.h>
#include <thread>

#include "glutil.h"


class OpenGLManager
{
public:
	static OpenGLManager& getInstance()
	{
		static OpenGLManager manager;
		return manager;
	}

	void init(int argc, char** argv, int width, int height)
	{
		this->running = true;
		this->runThread = std::thread(&OpenGLManager::loop, this, argc, argv, width, height);
		this->runThread.detach();
	}

	void close()
	{

	}
	void draw()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glEnable(GL_DEPTH_TEST);
		glShadeModel(GL_SMOOTH);
		glCullFace(GL_BACK);
		glEnable(GL_CULL_FACE);

		glFlush();
		glutSwapBuffers();

		GL_CHECK_ERRORS();
	}
	void reshape(int width, int height)
	{
		glViewport(0, 0, width, height);
	}

	void stop()
	{
		this->running = false;
	}

private:
	static void init_gl(int argc, char** argv, int width, int height);
	static void closeCallback() { OpenGLManager::getInstance().close(); }
	static void drawCallback() { OpenGLManager::getInstance().draw(); }
	static void reshapeCallback(int width, int height) { OpenGLManager::getInstance().reshape(width, height); }

	void loop(int argc, char** argv, int width, int height)
	{
		this->init_gl(argc, argv, width, height);

		while (this->running)
		{
			this->refresh();
		}
	}

	void refresh()
	{
		glutMainLoopEvent();
		glutPostRedisplay();
	}

	bool running = false;
	std::thread runThread;
};
