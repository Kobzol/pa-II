////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	sceneSetting.h
//
// summary:	Declaration and implementation of SceneSetting class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __SCENESETTING_H_
#define __SCENESETTING_H_

#include <dynArray.h>
#include <singletonDestroyer.h>

#pragma region FORWARD DECLARATIONS
class Camera;
class ShaderProgram;
struct Light;
#pragma endregion

typedef enum
{
	NONE = 0,
	WIREFRANE = 1,
	SOLID = 2,
	CULLING = 4,
	SHADOWS = 8
} e_DrawingMode;


class SceneSetting
{
private:
	static SceneSetting *SINGLETON_INSTANCE;
	static SingletonDestroyer<SceneSetting> destroyer;

public:
	e_DrawingMode m_drawingMode;
	unsigned int m_screen[2];
	float  m_clearColor[4];
	Camera *m_activeCamera;
	ShaderProgram *m_activeShader;

private:
	SceneSetting() : m_drawingMode(e_DrawingMode::SOLID), m_activeCamera(nullptr), m_activeShader(nullptr)
	{
		m_screen[0] = m_screen[1] = 0;
		m_clearColor[0] = m_clearColor[1] = m_clearColor[2] = m_clearColor[3] = 0.0f;
	}

public:
	~SceneSetting(){}

	static SceneSetting* GetInstance()
	{
		if (SINGLETON_INSTANCE == NULL)
		{
			SINGLETON_INSTANCE = new SceneSetting();
			destroyer.SetDoomed(SINGLETON_INSTANCE);
		}
		return SINGLETON_INSTANCE;
	}
};


#endif
