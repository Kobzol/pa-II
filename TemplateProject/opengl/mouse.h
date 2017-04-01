////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	mouse.h
//
// summary:	Declaration and implementation of Mouse class (GLUT handler only)
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __MOUSE_H_
#define __MOUSE_H_

#include "camera.h"
#include <AntTweakBar.h>
#include "sceneSetting.h"

#define CAMERA_ZOOMING		0x01
#define CAMERA_MOVING		0x02
#define CAMERA_ROTATING		0x04

class Mouse 
{
private:
	int m_specialKey;
	SceneSetting *m_sceneSetting;

public:
	static Mouse* CALLBACK_INSTANCE;
	float m_startDrag[2];
	float m_lastPosition[2];
	float m_delta[2];
	bool clickPending = false;

	Camera*		m_camera;
	int			m_state;

	inline Mouse(void) : m_camera(0), m_state(0), m_sceneSetting(SceneSetting::GetInstance()){};
	inline Mouse(Camera* cam) : m_camera(cam), m_state(0), m_sceneSetting(SceneSetting::GetInstance()){};
	inline ~Mouse(void){};

	void activeMove(int x, int y);	
	void passiveMove(int x, int y);	
	void click(int button, int buttonState, int x, int y);
	void wheel(int, int, int x, int y);

	static void activeMoveCB(int x, int y) 
	{ 
		if (TwEventMouseMotionGLUT(x,y)) return;
		CALLBACK_INSTANCE->activeMove(x,y); 
	}
	
	static void passiveMoveCB(int x, int y) 
	{
		if (TwEventMouseMotionGLUT(x,y)) return;
		CALLBACK_INSTANCE->passiveMove(x,y); 
	}

	static void clickCB(int button, int buttonState, int x, int y) 
	{ 
		if (TwEventMouseButtonGLUT(button, buttonState, x,y)) return;
		CALLBACK_INSTANCE->click(button, buttonState, x, y); 
	}

	static void wheelCB(int button, int buttonState, int x, int y) 
	{ 
		CALLBACK_INSTANCE->wheel(button, buttonState, x, y); 
	}

};


#endif