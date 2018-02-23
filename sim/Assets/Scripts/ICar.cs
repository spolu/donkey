using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface ICar
{
	// Inputs require 0:1 input except steering which is -1:1, where 0 is center.
	void RequestThrottle(float val);
	void RequestSteering(float val);
	void RequestFootBrake(float val);
	void RequestHandBrake(float val);

	// Query
	float GetSteering();
	float GetThrottle();
	float GetFootBrake();
	float GetHandBrake();

	// Query state.
	Transform GetTransform();
	Vector3 GetPosition();
	Vector3 GetVelocity();
	Vector3 GetAccelleration();

	// Save and restore State
	void SavePosRot();
	void RestorePosRot();
}
