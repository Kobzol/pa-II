////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	sceneManager.h
//
// summary:	Declaration and implementation of GUIManager class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __GUIMANAGER_H_
#define __GUIMANAGER_H_

#include <singletonDestroyer.h>
#include <dynArray.h>
#include <AntTweakBar.h>

#include "sceneData.h"

class GUIManager
{
private:
	static GUIManager *SINGLETON_INSTANCE;
	static SingletonDestroyer<GUIManager> destroyer; 

public:
	DynArray<TwBar*> m_bars;
	SceneData *m_sceneData;

private: 
	GUIManager() : m_bars(DynArray<TwBar*>(8)), m_sceneData(nullptr) {};

public:
	~GUIManager(){};

	static GUIManager* getInstance()
	{
		if (SINGLETON_INSTANCE == NULL)  // is it the first call?
		{
			SINGLETON_INSTANCE = new GUIManager(); // create sole instance
			destroyer.SetDoomed(SINGLETON_INSTANCE);
		}

		return SINGLETON_INSTANCE;
	}

	inline void setSceneData(SceneData* sceneData) { m_sceneData = sceneData; }

	inline void init()
	{
		SceneSetting *ss = SceneSetting::GetInstance();

		// init TW
		TwInit(TW_OPENGL, NULL);
		TwGLUTModifiersFunc(glutGetModifiers);
		TwWindowSize(ss->m_screen[0], ss->m_screen[1]);

		TwDefine(" GLOBAL help='This example shows how to integrate AntTweakBar with GLUT and OpenGL.' ");
	}

	inline void display()
	{
		for (unsigned int i = 0; i<m_bars.size(); i++)
			TwRefreshBar(m_bars[i]);
		TwDraw();
	}

	inline void reshape()
	{
		SceneSetting *ss = SceneSetting::GetInstance();
		TwWindowSize(ss->m_screen[0], ss->m_screen[1]);
	}

	inline void free()
	{
		TwDeleteAllBars();
		TwTerminate();
	}

	inline void addBar(TwBar* bar){ if (bar) m_bars.push_back(bar); }
};

#endif