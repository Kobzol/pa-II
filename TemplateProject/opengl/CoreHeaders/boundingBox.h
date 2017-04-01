////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	boundingBox.h
//
// summary:	Declaration and implementation of BoundingBox class
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __BOUNDINGBOX_H_
#define __BOUNDINGBOX_H_

#include <glm.hpp>
#include "vcDefs.h"

enum BBOverlap
{
	BBO_NOOVERLAP	= 0,    //Element is completely in boundingbox
	BBO_INTERSECTED,
	BBO_AinB,
	BBO_BinA,
	BBO_IDENTICAL,
	BBO_UNDEFINED
};

class BoundingBox
{

public:
	//Fields
	bool m_isSet;
	glm::vec3 m_minP;
	glm::vec3 m_maxP;

	//constructors
	inline BoundingBox(void) : m_minP(glm::vec3(0.0f,0.0f,0.0f)), m_maxP(glm::vec3(0.0f,0.0f,0.0f)), m_isSet(false) {} //see metho IsNull
	inline BoundingBox(const BoundingBox &other) : m_minP(other.m_minP), m_maxP(other.m_maxP), m_isSet(true) {}
	inline BoundingBox(const glm::vec3 &commonMin) : m_minP(commonMin), m_maxP(commonMin), m_isSet(true) {}
	inline BoundingBox(const glm::vec3 &_minP, const glm::vec3 &_maxP) : m_minP(_minP), m_maxP(_maxP), m_isSet(true) {}

	inline ~BoundingBox(void){};

	friend BoundingBox operator+(const BoundingBox &a, const BoundingBox &b)
	{
		return BoundingBox( glm::vec3(MINIMUM(a.m_minP.x,b.m_minP.x),MINIMUM(a.m_minP.y,b.m_minP.y), MINIMUM(a.m_minP.z,b.m_minP.z)), glm::vec3(MAXIMUM(a.m_maxP.x,b.m_maxP.x),MAXIMUM(a.m_maxP.y,b.m_maxP.y), MAXIMUM(a.m_maxP.z,b.m_maxP.z)));
	}

	inline BoundingBox& operator+=(const BoundingBox &a)
	{
		if ((m_isSet)&&(a.m_isSet))
		{
			m_minP.x = MINIMUM(m_minP.x, a.m_minP.x);
			m_minP.y = MINIMUM(m_minP.y, a.m_minP.y);
			m_minP.z = MINIMUM(m_minP.z, a.m_minP.z);

			m_maxP.x = MAXIMUM(m_maxP.x, a.m_maxP.x);
			m_maxP.y = MAXIMUM(m_maxP.y, a.m_maxP.y);
			m_maxP.z = MAXIMUM(m_maxP.z, a.m_maxP.z);
		}
		return *this;
	}

	//Methods
	inline float Diagonal(){ glm::distance(m_minP, m_maxP); }
	
	void Center(float *point);

	void Set(const glm::vec3 &p );
	void Set(const float x, const float y, const float z );

	void AddPoint(const glm::vec3 &point);
	void AddPoint(float x, float y, float z);

	bool isIn(const glm::vec3 &p) const;

	BBOverlap overlapTest(const BoundingBox &other) const;
	bool intersectPlane( const glm::vec3& normal, const float distanceFromOrigin ) const;
};

inline bool BoundingBox::isIn(const glm::vec3 &p) const
{
	return ((p.x >= m_minP.x) && (p.x <= m_maxP.x) && (p.y >= m_minP.y) && (p.y <= m_maxP.y) &&(p.z >= m_minP.z) && (p.z <= m_maxP.z));
}

inline void BoundingBox::Set( const glm::vec3 &point )
{
	m_minP = m_maxP = point;
	m_isSet = true;
}

inline void BoundingBox::Set( const float x, const float y, const float z)
{
	m_minP.x = m_maxP.x = x;
	m_minP.y = m_maxP.y = y;
	m_minP.z = m_maxP.z = z;
	m_isSet = true;
}

inline void BoundingBox::AddPoint(const glm::vec3 &point)
{
	if(!m_isSet) Set(point);
	else 
	{
		if(m_minP.x > point.x) m_minP.x = point.x;
		if(m_minP.y > point.y) m_minP.y = point.y;
		if(m_minP.z > point.z) m_minP.z = point.z;

		if(m_maxP.x < point.x) m_maxP.x = point.x;
		if(m_maxP.y < point.y) m_maxP.y = point.y;
		if(m_maxP.z < point.z) m_maxP.z = point.z;
	}
}

inline void BoundingBox::AddPoint(const float x, const float y, const float z)
{
	if(!m_isSet) Set(x,y,z);
	else 
	{
		if(m_minP.x > x) m_minP.x = x;
		if(m_minP.y > y) m_minP.y = y;
		if(m_minP.z > z) m_minP.z = z;

		if(m_maxP.x < x) m_maxP.x = x;
		if(m_maxP.y < y) m_maxP.y = y;
		if(m_maxP.z < z) m_maxP.z = z;
	}
}

inline void BoundingBox::Center(float *point)
{
	point[0] = 0.5f * (m_maxP.x + m_minP.x);
	point[1] = 0.5f * (m_maxP.y + m_minP.y);
	point[2] = 0.5f * (m_maxP.z + m_minP.z);
}

inline BBOverlap BoundingBox::overlapTest(const BoundingBox &other) const
{
	if ((m_maxP == other.m_maxP)&&(m_minP == other.m_minP)) return BBO_IDENTICAL; 

	glm::vec3 CA(0.5f * (m_maxP.x + m_minP.x), 0.5f * (m_maxP.y + m_minP.y), 0.5f * (m_maxP.z + m_minP.z));
	glm::vec3 CAext = CA - m_minP;

	glm::vec3 CB(0.5f * (other.m_maxP.x + other.m_minP.x), 0.5f * (other.m_maxP.y + other.m_minP.y), 0.5f * (other.m_maxP.z + other.m_minP.z));
	glm::vec3 CBext = CB - other.m_minP;

	glm::vec3  T = CB - CA;
	if ((abs(T.x) < (CAext.x + CBext.x)) && (abs(T.y) < (CAext.y + CBext.y)) && (abs(T.z) < (CAext.z + CBext.z)))
	{
		//Some kind of overlapping
		bool minBInA = ((other.m_minP.x >= m_minP.x) && (other.m_minP.x <= m_maxP.x) && (other.m_minP.y >= m_minP.y) && (other.m_minP.y <= m_maxP.y) &&(other.m_minP.z >= m_minP.z) && (other.m_minP.z <= m_maxP.z));
		bool maxBinA = ((other.m_maxP.x >= m_minP.x) && (other.m_maxP.x <= m_maxP.x) && (other.m_maxP.y >= m_minP.y) && (other.m_maxP.y <= m_maxP.y) &&(other.m_maxP.z >= m_minP.z) && (other.m_maxP.z <= m_maxP.z));

		if (minBInA && maxBinA)
			return BBO_BinA;

		if (minBInA || maxBinA)
			return BBO_INTERSECTED;

		return BBO_AinB;
	}
	else
		return BBO_NOOVERLAP;
}

inline bool BoundingBox::intersectPlane( const glm::vec3 &normal, const float originOffset ) const
{
	glm::vec3 diagMin, diagMax;

	if( normal.x >= 0.0 )
	{
		diagMin.x = m_minP.x;
		diagMax.x = m_maxP.x;
	}
	else
	{
		diagMin.x = m_maxP.x;
		diagMax.x = m_minP.x;
	}

	if( normal.y >= 0.0 )
	{
		diagMin.y = m_minP.y;
		diagMax.y = m_maxP.y;
	}
	else
	{
		diagMin.y = m_maxP.y;
		diagMax.y = m_minP.y;
	}

	if( normal.z >= 0.0 )
	{
		diagMin.z = m_minP.z;
		diagMax.z = m_maxP.z;
	}
	else
	{
		diagMin.z = m_maxP.z;
		diagMax.z = m_minP.z;
	}

	float test = glm::dot(normal,diagMin) + originOffset;
	//if( test > 0.0 )
	if( test > FLOAT_EPSILON )
		return false;

	test = glm::dot(normal,diagMax) + originOffset;
	if (abs(test)<FLOAT_EPSILON)
		test=0.0f;

	if( test >= 0.0f )
		return true;
	else
		return false;
}

#endif