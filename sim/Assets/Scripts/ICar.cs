using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface ICar
{
	// Inputs require 0-1 input except steering which is in degrees, where 0 is center.
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

	// Mark the current activity for partial selections when creating training sets later.
	string GetActivity();
	void SetActivity(string act);

	// Save and restore State
	void SavePosRot();
	void RestorePosRot();
    void SetMaxSteering(float val);
    float GetMaxSteering();
}
