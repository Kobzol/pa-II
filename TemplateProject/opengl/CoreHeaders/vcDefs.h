////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	vcDefs.h
//
// summary:	Declaration and implementation of global basic definitions for the current application
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __SV_DEFS_H_
#define __SV_DEFS_H_

#define DllExport   __declspec( dllexport )

#ifdef _MSC_VER
#define SMALLSIZE : unsigned char
#else
#define SMALLSIZE

#endif 
#define testBit(bit, state) ((state & bit ) == bit)

#define FLOAT_EPSILON 0.0001

#define MINIMUM(a, b) ((a) < (b) ? (a) : (b))
#define MAXIMUM(a, b) ((a) > (b) ? (a) : (b))
#define INTERPOLATE(first,last,x) ((x-first)/(last-first))
#define ISININTERVAL(first,last,x) ((first<=x)&&(x<=last))
#define NOTININTERVAL(first,last,x) ((x<first)||(last<x))

#define SAFE_DELETE(p) if(p){delete (p);(p)=0;}
#define SAFE_DELETE_ARRAY(p) if(p){delete[] (p);(p)=0;}

#endif