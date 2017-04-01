////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	SceneGUI.h
//
// summary:	Declaration and implementation of SceneGUI class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __SCENEGUI_H_
#define __SCENEGUI_H_

#include <AntTweakBar.h>
#include <freeglut.h>

extern double boidsSeparationFactor;
extern double boidsCohesionFactor;
extern double boidsAlignmentFactor;
extern double boidsGoalFactor;

extern double boidsSeparationNeighbourhood;
extern double boidsCohesionNeighbourhood;
extern double boidsAlignmentNeighbourhood;

extern double boidsMaxVelocity;
extern double boidsViewAngle;

extern double boidTestDir[3];

static class SceneGUI
{
public:
	static inline TwBar* createBar();

private:
	static inline void TW_CALL toggleFullscreen(void* tw_satisfy);
} _SceneGUI;

inline void TW_CALL SceneGUI::toggleFullscreen(void* tw_satisfy)
{ 
	glutFullScreenToggle();
}

static void TW_CALL scatter(void* tw_satisfy)
{
	boidsCohesionFactor = -boidsCohesionFactor;
}

inline TwBar* SceneGUI::createBar()
{
	TwBar* bar = TwNewBar("sceneBar");
	TwDefine("sceneBar                 "
		"size          = '200 300'     "
		"position      = '20 20'      "
		"color         = '0 0 0'  "
		"alpha         = 50           "
		"label         = 'Scene'  "
		"resizable     = False         "
		"fontresizable = True         "
		"iconifiable   = True          ");

	TwAddButton(bar, "Fullscreen", toggleFullscreen, NULL,
		"group = 'Screen' "
		"label = 'Toggle Fullscreen' "
		"help  = 'Toggle Fullscreen' ");
	
	TwAddVarRW(bar, "sepFactor", TW_TYPE_DOUBLE, &boidsSeparationFactor,
		" label='Separation factor' min=-4 max=4 step=0.1 keyIncr=a keyDecr=A help='Separation factor' ");
	TwAddVarRW(bar, "cohFactor", TW_TYPE_DOUBLE, &boidsCohesionFactor,
		" label='Cohesion factor' min=-4 max=4 step=0.1 keyIncr=d keyDecr=D help='Cohesion factor' ");
	TwAddVarRW(bar, "aliFactor", TW_TYPE_DOUBLE, &boidsAlignmentFactor,
		" label='Alignment factor' min=-4 max=4 step=0.1 keyIncr=d keyDecr=F help='Alignment factor' ");
	TwAddVarRW(bar, "goalFactor", TW_TYPE_DOUBLE, &boidsGoalFactor,
		" label='Goal factor' min=-4 max=4 step=0.01 keyIncr=g keyDecr=G help='Goal factor' ");

	TwAddVarRW(bar, "sepNeigh", TW_TYPE_DOUBLE, &boidsSeparationNeighbourhood,
		" label='Separation neighbourhood' min=-4 max=4 step=0.1 keyIncr=c keyDecr=C help='Separation neighbourhood' ");
	TwAddVarRW(bar, "cohNeigh", TW_TYPE_DOUBLE, &boidsCohesionNeighbourhood,
		" label='Cohesion neighbourhood' min=-4 max=4 step=0.1 keyIncr=v keyDecr=V help='Cohesion neighbourhood' ");
	TwAddVarRW(bar, "aliNeigh", TW_TYPE_DOUBLE, &boidsAlignmentNeighbourhood,
		" label='Alignment neighbourhood' min=-4 max=4 step=0.01 keyIncr=b keyDecr=B help='Alignment neighbourhood' ");

	TwAddVarRW(bar, "maxVelocity", TW_TYPE_DOUBLE, &boidsMaxVelocity,
		" label='Max velocity' min=0 max=1 step=0.01 keyIncr=x keyDecr=X help='Max velocity' ");
	TwAddVarRW(bar, "viewAngle", TW_TYPE_DOUBLE, &boidsViewAngle,
		" label='View angle' min=0 max=180 step=1.0 keyIncr=h keyDecr=H help='View angle' ");

	TwAddVarRW(bar, "testRotation", TW_TYPE_DIR3D, &boidTestDir,
		" label='Test rotation' ");

	TwAddButton(bar, "Scatter", scatter, NULL,
		"group = 'Screen' "
		"label = 'Scatter' key=s "
		"help  = 'Scatter' ");

	return bar;
}
#endif
