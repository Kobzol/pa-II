#include "opengl.h"

#include <cstdio>

#include <glew.h>
#include <freeglut.h>

#include "glutil.h"


void OpenGLManager::init_gl(int argc, char** argv, int width, int height)
{
	glutInit(&argc, argv);

	//glutInitContextVersion(4, 4);
	glutInitContextProfile(GLUT_CORE_PROFILE);						//glutInitContextProfile(GLUT_COMPATIBILITY_PROFILE);
																	//glutInitContextFlags(GLUT_FORWARD_COMPATIBLE | GLUT_DEBUG);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(0, 0);

	int window = glutCreateWindow(0);
	
	char textBuffer[512];
	sprintf_s(textBuffer, 512, "SimpleView | context %s | renderer %s | vendor %s ",
		(const char*)glGetString(GL_VERSION),
		(const char*)glGetString(GL_RENDERER),
		(const char*)glGetString(GL_VENDOR));
	glutSetWindowTitle(textBuffer);

	glewExperimental = TRUE;
	if (GLEW_OK != glewInit())
	{
		exit(1);
	}
	if (!glewIsSupported("GL_VERSION_3_3 "))
	{
		printf("ERROR: Support for necessary OpenGL extensions missing.");
		return;
	}

	GL_CHECK_ERRORS();
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_EXIT);

	//Close function
	glutCloseFunc(OpenGLManager::closeCallback);

	// Display and Idle
	glutDisplayFunc(OpenGLManager::drawCallback);
	glutReshapeFunc(OpenGLManager::reshapeCallback);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}
