////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	mouse.cpp
//
// summary:	Implements the mouse class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "mouse.h"
#include "sceneManager.h"

#if !defined(GLUT_WHEEL_UP)
#  define GLUT_WHEEL_UP   3
#  define GLUT_WHEEL_DOWN 4
#endif

Mouse* Mouse::CALLBACK_INSTANCE = NULL;

void Mouse::activeMove(int x, int y) 
{
	m_delta[0] = (x - m_lastPosition[0]) * 0.25f;
	m_delta[1] = (y - m_lastPosition[1]) * 0.25f;

	m_lastPosition[0] = (float)x;
	m_lastPosition[1] = (float)y;

	if ( testBit(CAMERA_ZOOMING, m_state) ) 
	{
		m_camera->zoom(m_delta[1], m_camera->getOrbitMinZoom(), m_camera->getOrbitMaxZoom(), 
					m_sceneSetting->m_screen[0],
					m_sceneSetting->m_screen[1]);
	}

	if ( testBit(CAMERA_ROTATING, m_state) ) 
	{
		m_camera->rotateSmoothly(m_delta[0], m_delta[1], 0.0f);
	}

	if ( testBit(CAMERA_MOVING, m_state ) ) 
	{
		printf("(%f, %f, %f)\n", m_delta[0], m_delta[1], 0.0f);
		m_camera->shift(m_delta[0], m_delta[1], 0.0f);
	}
	SceneManager::GetInstance()->Redisplay();
}

void Mouse::passiveMove(int x, int y) 
{
	//SceneManager::GetInstance()->Pick(x,y);
	//SceneManager::GetInstance()->Redisplay();
}

void Mouse::click(int button, int buttonState, int x, int y) 
{
	m_specialKey = glutGetModifiers();
	switch ( button ) 
	{
		case GLUT_RIGHT_BUTTON:
			switch (m_specialKey)
			{
				case GLUT_ACTIVE_CTRL:
					m_state = CAMERA_ROTATING;
					break;
				case GLUT_ACTIVE_ALT:
					m_state = CAMERA_ZOOMING;
					break;
				case GLUT_ACTIVE_SHIFT:
					m_state = CAMERA_MOVING;
					break;
				default:
					m_state = CAMERA_ROTATING;
					break;
			}
			break;
		case GLUT_MIDDLE_BUTTON:
			//state = CAMERA_ZOOMING;
			break;
		case GLUT_LEFT_BUTTON:
			//state = CAMERA_MOVING;
			break;
		case GLUT_WHEEL_UP:
			//state = CAMERA_ZOOMING;
			break;
		case GLUT_WHEEL_DOWN:
			//state = CAMERA_ZOOMING;
			break;

	}
		
	if ( buttonState == GLUT_DOWN) 
	{
		m_startDrag[0] = (float)x;
		m_startDrag[1] = (float)y;

		m_lastPosition[0] = (float)x;
		m_lastPosition[1] = (float)y;
	}
	else 
	{		
		m_state = 0;
	}

	if (buttonState == GLUT_DOWN && button == GLUT_LEFT_BUTTON)
	{
		this->clickPending = true;
	}
}

void Mouse::wheel(int button, int buttonState, int x, int y) 
{
	m_specialKey = glutGetModifiers();
	switch ( buttonState ) 
	{
		case 1: 
			m_camera->zoom(-50, m_camera->getOrbitMinZoom(), m_camera->getOrbitMaxZoom(), 
								m_sceneSetting->m_screen[0],
								m_sceneSetting->m_screen[1]);
			break;
		case -1:
			m_camera->zoom(50, m_camera->getOrbitMinZoom(), m_camera->getOrbitMaxZoom(), 
								m_sceneSetting->m_screen[0],
								m_sceneSetting->m_screen[1]);
			break;
	}
	m_state = 0;
}

