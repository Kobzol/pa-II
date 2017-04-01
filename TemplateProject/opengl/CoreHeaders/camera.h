////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	camera.h
//
// summary:	Declares and implements the camera class
//			Necessary updates because of GLM library.
//			All methods dealing with camera rotations were reimplemented because of different interpretation of
//			pitch, yaw and roll in the quaternion class. Moreover, GLM data classes and structures were used.
//			Corrected BUG:  m_mvpMatrix = m_mvMatrix * m_pMatrix;   ->    m_mvpMatrix = m_pMatrix * m_mvMatrix; !!!
//			Added normal matrix support.
//			All methods became inline because of easier class reusability in different libraries :-(
// author:	Petr Gajdoš
// 
// Copyright © 2014 Petr Gajdoš. All Rights Reserved for updates only.
// 
//-----------------------------------------------------------------------------
// Copyright (c) 2007-2008 dhpoware. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// A general purpose 6DoF (six degrees of freedom) quaternion based camera.
//
// This camera class supports 4 different behaviors:
// first person mode, spectator mode, flight mode, and orbit mode.
//
// First person mode only allows 5DOF (x axis movement, y axis movement, z axis
// movement, yaw, and pitch) and movement is always parallel to the world x-z
// (ground) plane.
//
// Spectator mode is similar to first person mode only movement is along the
// direction the camera is pointing.
// 
// Flight mode supports 6DoF. This is the camera class' default behavior.
//
// Orbit mode rotates the camera around a target position. This mode can be
// used to simulate a third person camera. Orbit mode supports 2 modes of
// operation: orbiting about the target's Y axis, and free orbiting. The former
// mode only allows pitch and yaw. All yaw changes are relative to the target's
// local Y axis. This means that camera yaw changes won't be affected by any
// rolling. The latter mode allows the camera to freely orbit the target. The
// camera is free to pitch, yaw, and roll. All yaw changes are relative to the
// camera's orientation (in space orbiting the target).
//
// This camera class allows the camera to be moved in 2 ways: using fixed
// step world units, and using a supplied velocity and acceleration. The former
// simply moves the camera by the specified amount. To move the camera in this
// way call one of the move() methods. The other way to move the camera
// calculates the camera's displacement based on the supplied velocity,
// acceleration, and elapsed time. To move the camera in this way call the
// updatePosition() method.
// 
//////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __CAMERA_H_
#define __CAMERA_H_

#define CAMERA_ZOOMING		0x01
#define CAMERA_MOVING		0x02
#define CAMERA_ROTATING		0x04

#define CAMERA_DEFAULT_SHIFT_SPEED 0.05f
#define CAMERA_DEFAULT_ZOOM_SPEED 0.01f;
#define CAMERA_DEFAULT_ROTATION_SPEED 0.3f
#define CAMERA_DEFAULT_FOVX 90.0f
#define CAMERA_DEFAULT_ZNEAR 0.01f
#define CAMERA_DEFAULT_ZFAR 500.0f
#define CAMERA_DEFAULT_ORBIT_MIN_ZOOM (CAMERA_DEFAULT_ZNEAR + 1.0f)
#define CAMERA_DEFAULT_ORBIT_MAX_ZOOM (CAMERA_DEFAULT_ZFAR * 0.5f)
#define CAMERA_DEFAULT_ORBIT_OFFSET_DISTANCE (CAMERA_DEFAULT_ORBIT_MIN_ZOOM + (CAMERA_DEFAULT_ORBIT_MAX_ZOOM - CAMERA_DEFAULT_ORBIT_MIN_ZOOM) * 0.25f)

#include <glm.hpp>
#include <gtc/quaternion.hpp>
#include <gtx/quaternion.hpp>
#include <gtc/matrix_transform.hpp> 
#include <gtc/matrix_inverse.hpp>
#include <gtx/norm.hpp>
#include <gtc/epsilon.hpp>

#include <math.h>
#include <algorithm>    // std::min

class Camera
{

public:
	enum CameraBehavior
	{
		CAMERA_BEHAVIOR_FIRST_PERSON,
		CAMERA_BEHAVIOR_SPECTATOR,
		CAMERA_BEHAVIOR_FLIGHT,
		CAMERA_BEHAVIOR_ORBIT
	};


private:
	CameraBehavior m_behavior;
	bool m_preferTargetUpOrbiting;
	float m_accumPitchDegrees;
	float m_savedAccumPitchDegrees;
	float m_zoomSpeed;
	float m_rotationSpeed;
	float m_fovx;
	float m_aspectRatio;
	float m_znear;
	float m_zfar;
	float m_orbitMinZoom;
	float m_orbitMaxZoom;
	float m_orbitOffsetDistance;
	float m_firstPersonYOffset;
	glm::vec3 m_eye;
	glm::vec3 m_savedEye;
	glm::vec3 m_target;
	glm::vec3 m_targetUp;
	glm::vec3 m_xAxis;
	glm::vec3 m_yAxis;
	glm::vec3 m_zAxis;
	glm::vec3 m_viewDir;
	glm::vec3 m_acceleration;
	glm::vec3 m_currentVelocity;
	glm::vec3 m_velocity;
	glm::quat m_orientation;
	glm::quat m_savedOrientation;
	glm::mat4 m_vMatrix;
	glm::mat4 m_pMatrix;
	glm::mat4 m_vpMatrix;
	glm::mat3 m_normalMatrix;

	bool m_isPerspectiveCamera;

	const glm::vec3 CAMERA_DEFAULT_WORLD_XAXIS;
	const glm::vec3 CAMERA_DEFAULT_WORLD_YAXIS;
	const glm::vec3 CAMERA_DEFAULT_WORLD_ZAXIS;


public:
    Camera();
	~Camera(){};

	inline void lookAt(const glm::vec3 &target) { lookAt(m_eye, target, m_yAxis); }
    void lookAt(const glm::vec3 &eye, const glm::vec3 &target, const glm::vec3 &up);

    void perspective(float fovx, float aspect, float znear, float zfar);
	void ortho(const float &left, const float &right, const float &bottom, const float &top, const float &zNear, const float &zFar);

    void move(float dx, float dy, float dz);
    void move(const glm::vec3 &direction, const glm::vec3 &amount);
    void rotate(float headingDegrees, float pitchDegrees, float rollDegrees);
    void rotateSmoothly(float headingDegrees, float pitchDegrees, float rollDegrees);
    void undoRoll();
    void updatePosition(const glm::vec3 &direction, float elapsedTimeSec);
    void zoom(float zoom, const float minZoom, const float maxZoom, const unsigned int screenWidth, const unsigned int screenHeight);
	void shift(float dx, float dy, float dz); 

    // Getter methods.
	inline const float getNearZ() const { return m_znear; }
	inline const float getFarZ() const { return m_zfar; }
	inline const float getFocusDistance() const { return (m_behavior == CAMERA_BEHAVIOR_ORBIT) ? m_orbitOffsetDistance : m_firstPersonYOffset; }
	inline const float getFOVX() const { return m_fovx; }
	inline const float getAspectRation() const { return m_aspectRatio; }

	inline const CameraBehavior getBehavior() const { return m_behavior; }
	inline const float getOrbitMinZoom() const { return m_orbitMinZoom; }
	inline const float getOrbitMaxZoom() const { return m_orbitMaxZoom; }
	inline const float getOrbitOffsetDistance() const { return m_orbitOffsetDistance; }
	inline const float getRotationSpeed() const { return m_rotationSpeed; }
	inline const glm::vec3 &getAcceleration() const { return m_acceleration; }
	inline const glm::vec3 &getCurrentVelocity() const { return m_currentVelocity; }
	inline const glm::vec3 &getPosition() const { return m_eye; }
	inline const glm::vec3 &getTarget() const { return m_target; }
	inline const glm::vec3 &getVelocity() const { return m_velocity; }
	inline const glm::vec3 &getViewDirection() const { return m_viewDir; }
	inline const glm::vec3 &getXAxis() const { return m_xAxis; }
	inline const glm::vec3 &getYAxis() const { return m_yAxis; }
	inline const glm::vec3 &getZAxis() const { return m_zAxis; }
	inline const glm::quat &getOrientation() const { return m_orientation; }
	inline const glm::mat4& getViewMatrix() const { return m_vMatrix; }
	inline const glm::mat4& getProjectionMatrix() const { return m_pMatrix; }
	inline const glm::mat4& getViewProjectionMatrix() const { return m_vpMatrix; }
	inline const glm::mat3& getNormalMatrix() const { return m_normalMatrix; }
	inline const bool preferTargetUpOrbiting() const { return m_preferTargetUpOrbiting; }
	inline const bool isPerspectiveCamera() const { return m_isPerspectiveCamera; }
    
	inline glm::mat4 &Camera::getPM(){ return m_pMatrix; }
	inline glm::mat4 &Camera::getVM(){ return m_vMatrix; }
	inline glm::mat4 &Camera::getVPM(){ return m_vpMatrix; }
	inline glm::mat3 &Camera::getNM(){ return m_normalMatrix; }

    // Setter methods.
	inline void setAcceleration(const glm::vec3 &acceleration) { m_acceleration = acceleration; }
    void setBehavior(CameraBehavior newBehavior);
	inline void setCurrentVelocity(const glm::vec3 &currentVelocity) { m_currentVelocity = currentVelocity; }
	inline void setCurrentVelocity(float x, float y, float z) { m_currentVelocity = glm::vec3(x, y, z); }
	inline void setOrbitMaxZoom(float orbitMaxZoom) { m_orbitMaxZoom = orbitMaxZoom; }
	inline void setOrbitMinZoom(float orbitMinZoom) { m_orbitMinZoom = orbitMinZoom; }
	inline void setOrbitOffsetDistance(float orbitOffsetDistance) { m_orbitOffsetDistance = orbitOffsetDistance; }
    void setOrientation(const glm::quat &newOrientation);
	inline void setPosition(const glm::vec3 &newEye) { m_eye = newEye; updateViewMatrix(); }
	inline void setTarget(const glm::vec3 &newTarget) { m_target = newTarget; updateViewMatrix(); }
	void setPreferTargetYAxisOrbiting(bool preferTargetYAxisOrbiting);
	inline void setRotationSpeed(float rotationSpeed) { m_rotationSpeed = rotationSpeed; }
	inline void setVelocity(const glm::vec3 &velocity) { m_velocity = velocity; }
	inline void setVelocity(float x, float y, float z) { m_velocity = glm::vec3(x, y, z); }

private:
    void rotateFirstPerson(float headingDegrees, float pitchDegrees);
    void rotateFlight(float headingDegrees, float pitchDegrees, float rollDegrees);
    void rotateOrbit(float headingDegrees, float pitchDegrees, float rollDegrees);
    void updateVelocity(const glm::vec3 &direction, float elapsedTimeSec);
    void updateViewMatrix();
};

//const glm::vec3 Camera::CAMERA_DEFAULT_WORLD_XAXIS(1.0f, 0.0f, 0.0f);
//const glm::vec3 Camera::CAMERA_DEFAULT_WORLD_YAXIS(0.0f, 1.0f, 0.0f);
//const glm::vec3 Camera::CAMERA_DEFAULT_WORLD_ZAXIS(0.0f, 0.0f, 1.0f);

inline Camera::Camera() : 
	CAMERA_DEFAULT_WORLD_XAXIS(glm::vec3(1.0f, 0.0f, 0.0f)), 
	CAMERA_DEFAULT_WORLD_YAXIS(glm::vec3(0.0f, 1.0f, 0.0f)), 
	CAMERA_DEFAULT_WORLD_ZAXIS(glm::vec3(0.0f, 0.0f, 1.0f))
{
	m_isPerspectiveCamera = true;

	m_behavior = CAMERA_BEHAVIOR_FLIGHT;
	m_preferTargetUpOrbiting = true;

	m_accumPitchDegrees = 0.0f;
	m_savedAccumPitchDegrees = 0.0f;

	m_zoomSpeed = CAMERA_DEFAULT_ZOOM_SPEED;
	m_rotationSpeed = CAMERA_DEFAULT_ROTATION_SPEED;
	m_fovx = CAMERA_DEFAULT_FOVX;
	m_aspectRatio = 0.0f;
	m_znear = CAMERA_DEFAULT_ZNEAR;
	m_zfar = CAMERA_DEFAULT_ZFAR;

	m_orbitMinZoom = CAMERA_DEFAULT_ORBIT_MIN_ZOOM;
	m_orbitMaxZoom = CAMERA_DEFAULT_ORBIT_MAX_ZOOM;
	m_orbitOffsetDistance = CAMERA_DEFAULT_ORBIT_OFFSET_DISTANCE;

	m_eye = glm::vec3(0.0f, 0.0f, 0.0f);
	m_savedEye = glm::vec3(0.0f, 0.0f, 0.0f);
	m_target = glm::vec3(0.0f, 0.0f, 0.0f);

	//m_targetUp = glm::vec3(0.0f, 1.0f, 0.0f);
	m_targetUp = CAMERA_DEFAULT_WORLD_ZAXIS;

	//m_xAxis = glm::vec3(1.0f, 0.0f, 0.0f);
	//m_yAxis = glm::vec3(0.0f, 0.0f, 1.0f);						//Y->Z  swap
	//m_zAxis = glm::vec3(0.0f, 1.0f, 0.0f);						//Z->Y	swap
	m_xAxis = CAMERA_DEFAULT_WORLD_XAXIS;
	m_yAxis = CAMERA_DEFAULT_WORLD_YAXIS;
	m_zAxis = CAMERA_DEFAULT_WORLD_ZAXIS;

	m_viewDir = glm::vec3(0.0f, 0.0f, -1.0f);

	m_acceleration = glm::vec3(0.0f, 0.0f, 0.0f);
	m_currentVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
	m_velocity = glm::vec3(0.0f, 0.0f, 0.0f);

	m_orientation = glm::quat();
	m_savedOrientation = glm::quat();

	m_vMatrix = glm::mat4(1.0f);
	m_pMatrix = glm::mat4(1.0f);
	m_vpMatrix = glm::mat4(1.0f);
	m_normalMatrix = glm::mat3(1.0f);
}

inline void Camera::lookAt(const glm::vec3 &eye, const glm::vec3 &target, const glm::vec3 &up)
{
	m_eye = eye;
	m_target = target;

	m_zAxis = eye - target;
	m_zAxis = glm::normalize(m_zAxis);

	m_viewDir = -m_zAxis;

	m_xAxis = glm::cross(up, m_zAxis);
	m_xAxis = glm::normalize(m_xAxis);

	m_yAxis = glm::cross(m_zAxis, m_xAxis);
	m_yAxis = glm::normalize(m_yAxis);

	m_vMatrix[0][0] = m_xAxis.x;
	m_vMatrix[1][0] = m_xAxis.y;
	m_vMatrix[2][0] = m_xAxis.z;
	m_vMatrix[3][0] = -glm::dot(m_xAxis, eye);

	m_vMatrix[0][1] = m_yAxis.x;
	m_vMatrix[1][1] = m_yAxis.y;
	m_vMatrix[2][1] = m_yAxis.z;
	m_vMatrix[3][1] = -glm::dot(m_yAxis, eye);

	m_vMatrix[0][2] = m_zAxis.x;
	m_vMatrix[1][2] = m_zAxis.y;
	m_vMatrix[2][2] = m_zAxis.z;
	m_vMatrix[3][2] = -glm::dot(m_zAxis, eye);

	// Extract the pitch angle from the view matrix.
	m_accumPitchDegrees = glm::degrees(asinf(m_vMatrix[1][2]));

	// m_orientation.fromMatrix(m_mvMatrix);
	//m_orientation = glm::quat_cast(m_mvMatrix);
	m_orientation = glm::toQuat(m_vMatrix);

	updateViewMatrix();
}

inline void Camera::perspective(float fovx, float aspect, float znear, float zfar)
{
	m_isPerspectiveCamera = true;

	// Construct a projection matrix based on the horizontal field of view
	// 'fovx' rather than the more traditional vertical field of view 'fovy'.
	float e = 1.0f / tanf(glm::radians(fovx) / 2.0f);
	float aspectInv = 1.0f / aspect;
	float fovy = 2.0f * atanf(aspectInv / e);
	float xScale = 1.0f / tanf(0.5f * fovy);
	float yScale = xScale / aspectInv;

	m_pMatrix[0][0] = xScale;
	m_pMatrix[0][1] = 0.0f;
	m_pMatrix[0][2] = 0.0f;
	m_pMatrix[0][3] = 0.0f;

	m_pMatrix[1][0] = 0.0f;
	m_pMatrix[1][1] = yScale;
	m_pMatrix[1][2] = 0.0f;
	m_pMatrix[1][3] = 0.0f;

	m_pMatrix[2][0] = 0.0f;
	m_pMatrix[2][1] = 0.0f;
	m_pMatrix[2][2] = (znear + zfar) / (znear - zfar);
	m_pMatrix[2][3] = -1.0f;

	m_pMatrix[3][0] = 0.0f;
	m_pMatrix[3][1] = 0.0f;
	m_pMatrix[3][2] = (2.0f * znear * zfar) / (znear - zfar);
	m_pMatrix[3][3] = 0.0f;

	m_vpMatrix = m_pMatrix * m_vMatrix;

	m_fovx = fovx;
	m_aspectRatio = aspect;
	m_znear = znear;
	m_zfar = zfar;
}

//left, right, bottom, top is set relatively to the camera target !!!
inline void Camera::ortho(const float &left, const float &right, const float &bottom, const float &top, const float &zNear, const float &zFar)
{
	m_isPerspectiveCamera = false;

	m_pMatrix = glm::ortho<float>(left, right, bottom, top, zNear, zFar);

	//m_pMatrix[0][0] = 2.0 / (right - left);
	//   m_pMatrix[0][1] = 0.0f;
	//   m_pMatrix[0][2] = 0.0f;
	//   m_pMatrix[0][3] = 0.0f;

	//   m_pMatrix[1][0] = 0.0f;
	//   m_pMatrix[1][1] = 2.0f / (top - bottom);
	//   m_pMatrix[1][2] = 0.0f;
	//   m_pMatrix[1][3] = 0.0f;

	//   m_pMatrix[2][0] = 0.0f;
	//   m_pMatrix[2][1] = 0.0f;
	//   m_pMatrix[2][2] = - 2.0f / (zFar - zNear);
	//   m_pMatrix[2][3] = 0.0f;

	//   m_pMatrix[3][0] = - (right + left) / (right - left);
	//   m_pMatrix[3][1] = - (top + bottom) / (top - bottom);
	//   m_pMatrix[3][2] = - (zFar + zNear) / (zFar - zNear);
	//   m_pMatrix[3][3] = 0.0f;

	m_vpMatrix = m_pMatrix * m_vMatrix;
}

inline void Camera::move(float dx, float dy, float dz)
{
	// Moves the camera by dx world units to the left or right; dy
	// world units upwards or downwards; and dz world units forwards
	// or backwards.

	if (m_behavior == CAMERA_BEHAVIOR_ORBIT)
	{
		// Orbiting camera is always positioned relative to the
		// target position. See updateViewMatrix().
		return;
	}

	glm::vec3 eye = m_eye;
	glm::vec3 forwards;

	if (m_behavior == CAMERA_BEHAVIOR_FIRST_PERSON)
	{
		// Calculate the forwards direction. Can't just use the camera's local
		// z axis as doing so will cause the camera to move more slowly as the
		// camera's view approaches 90 degrees straight up and down.

		forwards = glm::cross(CAMERA_DEFAULT_WORLD_YAXIS, m_xAxis);
		forwards = glm::normalize(forwards);
	}
	else
	{
		forwards = m_viewDir;
	}

	eye += m_xAxis * dx;
	eye += CAMERA_DEFAULT_WORLD_YAXIS * dy;
	eye += forwards * dz;

	setPosition(eye);
}

inline void Camera::move(const glm::vec3 &direction, const glm::vec3 &amount)
{
	// Moves the camera by the specified amount of world units in the specified
	// direction in world space.

	if (m_behavior == CAMERA_BEHAVIOR_ORBIT)
	{
		// Orbiting camera is always positioned relative to the
		// target position. See updateViewMatrix().
		return;
	}

	m_eye.x += direction.x * amount.x;
	m_eye.y += direction.y * amount.y;
	m_eye.z += direction.z * amount.z;

	updateViewMatrix();
}

inline void Camera::rotate(float headingDegrees, float pitchDegrees, float rollDegrees)
{
	// Rotates the camera based on its current behavior.
	// Note that not all behaviors support rolling.

	pitchDegrees = -pitchDegrees;
	headingDegrees = -headingDegrees;
	rollDegrees = -rollDegrees;

	switch (m_behavior)
	{
	default:
		break;

	case CAMERA_BEHAVIOR_FIRST_PERSON:
	case CAMERA_BEHAVIOR_SPECTATOR:
		rotateFirstPerson(headingDegrees, pitchDegrees);
		break;

	case CAMERA_BEHAVIOR_FLIGHT:
		rotateFlight(headingDegrees, pitchDegrees, rollDegrees);
		break;

	case CAMERA_BEHAVIOR_ORBIT:
		rotateOrbit(headingDegrees, pitchDegrees, rollDegrees);
		break;
	}

	updateViewMatrix();
}

inline void Camera::rotateSmoothly(float headingDegrees, float pitchDegrees, float rollDegrees)
{
	// This method applies a scaling factor to the rotation angles prior to
	// using these rotation angles to rotate the camera. This method is usually
	// called when the camera is being rotated using an input device (such as a
	// mouse or a joystick). 

	headingDegrees *= m_rotationSpeed;
	pitchDegrees *= m_rotationSpeed;
	rollDegrees *= m_rotationSpeed;

	rotate(headingDegrees, pitchDegrees, rollDegrees);
}

inline void Camera::undoRoll()
{
	// Undo any camera rolling by leveling the camera. When the camera is
	// orbiting this method will cause the camera to become level with the
	// orbit target.

	if (m_behavior == CAMERA_BEHAVIOR_ORBIT)
		lookAt(m_eye, m_target, m_targetUp);
	else
		lookAt(m_eye, m_eye + m_viewDir, CAMERA_DEFAULT_WORLD_YAXIS);
}

inline void Camera::updatePosition(const glm::vec3 &direction, float elapsedTimeSec)
{
	// Moves the camera using Newton's second law of motion. Unit mass is
	// assumed here to somewhat simplify the calculations. The direction vector
	// is in the range [-1,1].

	if (glm::length2(m_currentVelocity) != 0.0f)
	{
		// Only move the camera if the velocity vector is not of zero length.
		// Doing this guards against the camera slowly creeping around due to
		// floating point rounding errors.

		glm::vec3 displacement = (m_currentVelocity * elapsedTimeSec) +
			(0.5f * m_acceleration * elapsedTimeSec * elapsedTimeSec);

		// Floating point rounding errors will slowly accumulate and cause the
		// camera to move along each axis. To prevent any unintended movement
		// the displacement vector is clamped to zero for each direction that
		// the camera isn't moving in. Note that the updateVelocity() method
		// will slowly decelerate the camera's velocity back to a stationary
		// state when the camera is no longer moving along that direction. To
		// account for this the camera's current velocity is also checked.

		if (direction.x == 0.0f && glm::epsilonEqual(m_currentVelocity.x, 0.0f, 0.00001f))
			displacement.x = 0.0f;

		if (direction.y == 0.0f && glm::epsilonEqual(m_currentVelocity.y, 0.0f, 0.00001f))
			displacement.y = 0.0f;

		if (direction.z == 0.0f && glm::epsilonEqual(m_currentVelocity.z, 0.0f, 0.00001f))
			displacement.z = 0.0f;

		move(displacement.x, displacement.y, displacement.z);
	}

	// Continuously update the camera's velocity vector even if the camera
	// hasn't moved during this call. When the camera is no longer being moved
	// the camera is decelerating back to its stationary state.

	updateVelocity(direction, elapsedTimeSec);
}

inline void Camera::zoom(float zoom, const float minZoom, const float maxZoom, const unsigned int screenWidth, const unsigned int screenHeight)
{
	if (m_behavior == CAMERA_BEHAVIOR_ORBIT)
	{
		// Moves the camera closer to or further away from the orbit
		// target. The zoom amounts are in world units.

		m_orbitMaxZoom = maxZoom;
		m_orbitMinZoom = minZoom;

		glm::vec3 offset = m_eye - m_target;

		m_orbitOffsetDistance = glm::length(offset);
		offset = glm::normalize(offset);
		m_orbitOffsetDistance += zoom * m_zoomSpeed;
		m_orbitOffsetDistance = std::min(std::max(m_orbitOffsetDistance, minZoom), maxZoom);

		offset *= m_orbitOffsetDistance;
		m_eye = offset + m_target;

		updateViewMatrix();
	}
	else
	{
		// For the other behaviors zoom is interpreted as changing the
		// horizontal field of view. The zoom amounts refer to the horizontal
		// field of view in degrees.

		if (m_isPerspectiveCamera)
		{
			zoom = std::min(std::max(zoom * m_zoomSpeed, minZoom), maxZoom);
			perspective(zoom, m_aspectRatio, m_znear, m_zfar);
		}
		else
		{
			//m_pMatrix = glm::scale(m_pMatrix, glm::vec3(zoom, zoom, 1.0f));
			float zoomfactor = 1 + (zoom * 2.0f) / screenHeight;
			glm::vec3 zoomVector = glm::vec3(zoomfactor, zoomfactor, 1.0f);

			m_pMatrix[0] = m_pMatrix[0] * zoomVector[0];
			m_pMatrix[1] = m_pMatrix[1] * zoomVector[1];
			m_pMatrix[2] = m_pMatrix[2] * zoomVector[2];

			m_vpMatrix = m_pMatrix * m_vMatrix;
		}
	}
}

inline void Camera::shift(float dx, float dy, float dz)
{

	if (m_behavior == CAMERA_BEHAVIOR_ORBIT)
	{
		// Orbiting camera is always positioned relative to the
		// target position. See updateViewMatrix();

		// This makes a shift on XY plane based on camera orientation
		dx *= -1.0f;
		float realDX = m_vMatrix[0][0] * dx + m_vMatrix[0][1] * dy;
		float realDY = m_vMatrix[1][0] * dx + m_vMatrix[1][1] * dy;

		m_target.x += realDX * CAMERA_DEFAULT_SHIFT_SPEED;
		m_target.y += realDY * CAMERA_DEFAULT_SHIFT_SPEED;
		updateViewMatrix();
		return;
	}

	glm::vec3 eye = m_eye;
	glm::vec3 forwards;

	if (m_behavior == CAMERA_BEHAVIOR_FIRST_PERSON)
	{
		// Calculate the forwards direction. Can't just use the camera's local
		// z axis as doing so will cause the camera to move more slowly as the
		// camera's view approaches 90 degrees straight up and down.

		forwards = glm::cross(CAMERA_DEFAULT_WORLD_YAXIS, m_xAxis);
		forwards = glm::normalize(forwards);
	}
	else
	{
		forwards = m_viewDir;
	}

	eye += m_xAxis * dx * CAMERA_DEFAULT_SHIFT_SPEED;
	eye += CAMERA_DEFAULT_WORLD_YAXIS * dy * CAMERA_DEFAULT_SHIFT_SPEED;
	eye += forwards * dz * CAMERA_DEFAULT_SHIFT_SPEED;

	m_target += m_xAxis * dx * CAMERA_DEFAULT_SHIFT_SPEED;
	m_target += CAMERA_DEFAULT_WORLD_YAXIS * dy * CAMERA_DEFAULT_SHIFT_SPEED;
	m_target += forwards * dz * CAMERA_DEFAULT_SHIFT_SPEED;

	setPosition(eye);
}

inline void Camera::setBehavior(CameraBehavior newBehavior)
{
	// Switch to a new camera mode (i.e., behavior).
	// This method is complicated by the fact that it tries to save the current
	// behavior's state prior to making the switch to the new camera behavior.
	// Doing this allows seamless switching between camera behaviors.

	CameraBehavior prevBehavior = m_behavior;

	if (prevBehavior == newBehavior)
		return;

	m_behavior = newBehavior;
	return;

	switch (newBehavior)
	{
	case CAMERA_BEHAVIOR_FIRST_PERSON:
		switch (prevBehavior)
		{
		default:
			break;

		case CAMERA_BEHAVIOR_FLIGHT:
			m_eye.y = m_firstPersonYOffset;
			updateViewMatrix();
			break;

		case CAMERA_BEHAVIOR_SPECTATOR:
			m_eye.y = m_firstPersonYOffset;
			updateViewMatrix();
			break;

		case CAMERA_BEHAVIOR_ORBIT:
			m_eye.x = m_savedEye.x;
			m_eye.z = m_savedEye.z;
			m_eye.y = m_firstPersonYOffset;
			m_orientation = m_savedOrientation;
			m_accumPitchDegrees = m_savedAccumPitchDegrees;
			updateViewMatrix();
			break;
		}

		undoRoll();
		break;

	case CAMERA_BEHAVIOR_SPECTATOR:
		switch (prevBehavior)
		{
		default:
			break;

		case CAMERA_BEHAVIOR_FLIGHT:
			updateViewMatrix();
			break;

		case CAMERA_BEHAVIOR_ORBIT:
			m_eye = m_savedEye;
			m_orientation = m_savedOrientation;
			m_accumPitchDegrees = m_savedAccumPitchDegrees;
			updateViewMatrix();
			break;
		}

		undoRoll();
		break;

	case CAMERA_BEHAVIOR_FLIGHT:
		if (prevBehavior == CAMERA_BEHAVIOR_ORBIT)
		{
			m_eye = m_savedEye;
			m_orientation = m_savedOrientation;
			m_accumPitchDegrees = m_savedAccumPitchDegrees;
			updateViewMatrix();
		}
		else
		{
			m_savedEye = m_eye;
			updateViewMatrix();
		}
		break;

	case CAMERA_BEHAVIOR_ORBIT:
		if (prevBehavior == CAMERA_BEHAVIOR_FIRST_PERSON)
			m_firstPersonYOffset = m_eye.y;

		m_savedEye = m_eye;
		m_savedOrientation = m_orientation;
		m_savedAccumPitchDegrees = m_accumPitchDegrees;

		m_targetUp = CAMERA_DEFAULT_WORLD_ZAXIS;

		glm::vec3 newEye = m_eye + m_zAxis * m_orbitOffsetDistance;
		glm::vec3 newTarget = m_eye;

		lookAt(newEye, newTarget, m_targetUp);
		break;
	}
}

inline void Camera::setOrientation(const glm::quat &newOrientation)
{
	glm::mat4 m = glm::toMat4(newOrientation);

	// Store the pitch for this new orientation.
	// First person and spectator behaviors limit pitching to
	// 90 degrees straight up and down.

	m_accumPitchDegrees = glm::degrees(asinf(m[1][2]));

	// First person and spectator behaviors don't allow rolling.
	// Negate any rolling that might be encoded in the new orientation.

	m_orientation = newOrientation;

	if (m_behavior == CAMERA_BEHAVIOR_FIRST_PERSON || m_behavior == CAMERA_BEHAVIOR_SPECTATOR)
		lookAt(m_eye, m_eye + m_viewDir, CAMERA_DEFAULT_WORLD_YAXIS);

	updateViewMatrix();
}

inline void Camera::setPreferTargetYAxisOrbiting(bool preferTargetYAxisOrbiting)
{
	// Determines the behavior of Y axis rotations when the camera is
	// orbiting a target. When preferTargetYAxisOrbiting is true all
	// Y axis rotations are about the orbit target's local Y axis.
	// When preferTargetYAxisOrbiting is false then the camera's
	// local Y axis is used instead.

	m_preferTargetUpOrbiting = preferTargetYAxisOrbiting;

	if (m_preferTargetUpOrbiting)
		undoRoll();
}

inline void Camera::rotateFirstPerson(float headingDegrees, float pitchDegrees)
{
	// Implements the rotation logic for the first person style and
	// spectator style camera behaviors. Roll is ignored.

	m_accumPitchDegrees += pitchDegrees;

	if (m_accumPitchDegrees > 90.0f)
	{
		pitchDegrees = 90.0f - (m_accumPitchDegrees - pitchDegrees);
		m_accumPitchDegrees = 90.0f;
	}

	if (m_accumPitchDegrees < -90.0f)
	{
		pitchDegrees = -90.0f - (m_accumPitchDegrees - pitchDegrees);
		m_accumPitchDegrees = -90.0f;
	}

	glm::quat rot;

	if (pitchDegrees != 0.0f)
	{
		rot = glm::angleAxis(pitchDegrees, CAMERA_DEFAULT_WORLD_XAXIS);
		m_orientation = glm::normalize(rot * m_orientation);
	}
	if (headingDegrees != 0.0f)
	{
		rot = glm::angleAxis(headingDegrees, CAMERA_DEFAULT_WORLD_YAXIS);
		m_orientation = glm::normalize(m_orientation * rot);
	}
}

inline void Camera::rotateFlight(float headingDegrees, float pitchDegrees, float rollDegrees)
{
	// Implements the rotation logic for the flight style camera behavior.

	m_accumPitchDegrees += pitchDegrees;

	if (m_accumPitchDegrees > 360.0f)
		m_accumPitchDegrees -= 360.0f;

	if (m_accumPitchDegrees < -360.0f)
		m_accumPitchDegrees += 360.0f;

	glm::quat rot;

	//rot.fromHeadPitchRoll(headingDegrees, pitchDegrees, rollDegrees);
	//rot = glm::quat(glm::vec3(pitchDegrees, headingDegrees, rollDegrees));
	rot = glm::quat(glm::vec3(glm::radians(pitchDegrees), glm::radians(headingDegrees), glm::radians(rollDegrees)));

	m_orientation = glm::cross(m_orientation, rot);
}

inline void Camera::rotateOrbit(float headingDegrees, float pitchDegrees, float rollDegrees)
{
	// Implements the rotation logic for the orbit style camera behavior.
	// Roll is ignored for target Y axis orbiting.
	//
	// Briefly here's how this orbit camera implementation works. Switching to
	// the orbit camera behavior via the setBehavior() method will set the
	// camera's orientation to match the orbit target's orientation. Calls to
	// rotateOrbit() will rotate this orientation. To turn this into a third
	// person style view the updateViewMatrix() method will move the camera
	// position back 'm_orbitOffsetDistance' world units along the camera's
	// local z axis from the orbit target's world position.

	glm::quat rot;

	if (m_preferTargetUpOrbiting)
	{
		if (pitchDegrees != 0.0f)
		{
			rot = glm::angleAxis(pitchDegrees, CAMERA_DEFAULT_WORLD_XAXIS);
			m_orientation = glm::normalize(rot * m_orientation);
		}

		if (headingDegrees != 0.0f)
		{
			rot = glm::angleAxis(headingDegrees, m_targetUp);
			m_orientation = glm::normalize(m_orientation * rot);
		}

	}
	else
	{
		rot = glm::quat(glm::vec3(glm::radians(pitchDegrees), glm::radians(headingDegrees), glm::radians(rollDegrees)));
		m_orientation = glm::normalize(rot * m_orientation);
	}
}

inline void Camera::updateVelocity(const glm::vec3 &direction, float elapsedTimeSec)
{
	// Updates the camera's velocity based on the supplied movement direction
	// and the elapsed time (since this method was last called). The movement
	// direction is in the range [-1,1].

	if (direction.x != 0.0f)
	{
		// Camera is moving along the x axis.
		// Linearly accelerate up to the camera's max speed.

		m_currentVelocity.x += direction.x * m_acceleration.x * elapsedTimeSec;

		if (m_currentVelocity.x > m_velocity.x)
			m_currentVelocity.x = m_velocity.x;
		else if (m_currentVelocity.x < -m_velocity.x)
			m_currentVelocity.x = -m_velocity.x;
	}
	else
	{
		// Camera is no longer moving along the x axis.
		// Linearly decelerate back to stationary state.

		if (m_currentVelocity.x > 0.0f)
		{
			if ((m_currentVelocity.x -= m_acceleration.x * elapsedTimeSec) < 0.0f)
				m_currentVelocity.x = 0.0f;
		}
		else
		{
			if ((m_currentVelocity.x += m_acceleration.x * elapsedTimeSec) > 0.0f)
				m_currentVelocity.x = 0.0f;
		}
	}

	if (direction.y != 0.0f)
	{
		// Camera is moving along the y axis.
		// Linearly accelerate up to the camera's max speed.

		m_currentVelocity.y += direction.y * m_acceleration.y * elapsedTimeSec;

		if (m_currentVelocity.y > m_velocity.y)
			m_currentVelocity.y = m_velocity.y;
		else if (m_currentVelocity.y < -m_velocity.y)
			m_currentVelocity.y = -m_velocity.y;
	}
	else
	{
		// Camera is no longer moving along the y axis.
		// Linearly decelerate back to stationary state.

		if (m_currentVelocity.y > 0.0f)
		{
			if ((m_currentVelocity.y -= m_acceleration.y * elapsedTimeSec) < 0.0f)
				m_currentVelocity.y = 0.0f;
		}
		else
		{
			if ((m_currentVelocity.y += m_acceleration.y * elapsedTimeSec) > 0.0f)
				m_currentVelocity.y = 0.0f;
		}
	}

	if (direction.z != 0.0f)
	{
		// Camera is moving along the z axis.
		// Linearly accelerate up to the camera's max speed.

		m_currentVelocity.z += direction.z * m_acceleration.z * elapsedTimeSec;

		if (m_currentVelocity.z > m_velocity.z)
			m_currentVelocity.z = m_velocity.z;
		else if (m_currentVelocity.z < -m_velocity.z)
			m_currentVelocity.z = -m_velocity.z;
	}
	else
	{
		// Camera is no longer moving along the z axis.
		// Linearly decelerate back to stationary state.

		if (m_currentVelocity.z > 0.0f)
		{
			if ((m_currentVelocity.z -= m_acceleration.z * elapsedTimeSec) < 0.0f)
				m_currentVelocity.z = 0.0f;
		}
		else
		{
			if ((m_currentVelocity.z += m_acceleration.z * elapsedTimeSec) > 0.0f)
				m_currentVelocity.z = 0.0f;
		}
	}
}

inline void Camera::updateViewMatrix()
{
	// Reconstruct the view matrix.

	m_vMatrix = glm::toMat4(m_orientation);

	m_xAxis = glm::vec3(m_vMatrix[0][0], m_vMatrix[1][0], m_vMatrix[2][0]);
	m_yAxis = glm::vec3(m_vMatrix[0][1], m_vMatrix[1][1], m_vMatrix[2][1]);
	m_zAxis = glm::vec3(m_vMatrix[0][2], m_vMatrix[1][2], m_vMatrix[2][2]);
	m_viewDir = -m_zAxis;

	if (m_behavior == CAMERA_BEHAVIOR_ORBIT)
	{
		// Calculate the new camera position based on the current
		// orientation. The camera must always maintain the same
		// distance from the target. Use the current offset vector
		// to determine the correct distance from the target.

		m_eye = m_target + m_zAxis * m_orbitOffsetDistance;
	}

	m_vMatrix[3][0] = -glm::dot(m_xAxis, m_eye);
	m_vMatrix[3][1] = -glm::dot(m_yAxis, m_eye);
	m_vMatrix[3][2] = -glm::dot(m_zAxis, m_eye);

	m_vpMatrix = m_pMatrix * m_vMatrix;
	m_normalMatrix = glm::inverseTranspose(glm::mat3(m_vMatrix));	//The scale of m_mvMatrix does not need to be uniform !!!
	//m_normalMatrix = glm::mat3(m_mvMatrix);						//Only if m_mvMatrix scale is uniform !!!
}


#endif