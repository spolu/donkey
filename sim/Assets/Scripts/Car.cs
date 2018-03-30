using UnityEngine;
using System.Collections;


public class Car : MonoBehaviour, ICar {

	public WheelCollider[] wheelColliders;
	public Transform[] wheelMeshes;

	public float maxTorque = 50f;
	public float maxSpeed = 10f;
	public float maxSteer = 16.0f;

	public Transform centrOfMass;

	public float requestTorque = 0f;
	public float requestBrake = 0f;
	public float requestSteering = 0f;

	public Vector3 acceleration = Vector3.zero;
	public Vector3 prevVelocity = Vector3.zero;

	public Vector3 startPos;
	public Quaternion startRot;

	Rigidbody rb;

	//for logging
	public float lastSteer = 0.0f;
	public float lastAccel = 0.0f;

	// Use this for initialization
	void Awake () 
	{
		rb = GetComponent<Rigidbody>();

		if(rb && centrOfMass)
		{
			rb.centerOfMass = centrOfMass.localPosition;
		}

		requestTorque = 0f;
		requestSteering = 0f;
	}

	public void RequestThrottle(float val)
	{
		requestTorque = Mathf.Clamp(val, 0.0f, 1.0f);
	}

	public void RequestSteering(float val)
	{
		requestSteering = Mathf.Clamp(val, -1.0f, 1.0f);
	}

	public void RequestBrake(float val)
	{
		requestBrake = Mathf.Clamp(val, 0.0f, 1.0f);
	}

	public void Set(Vector3 pos, Quaternion rot)
	{
		rb.isKinematic = true;

		rb.position = pos;
		rb.rotation = rot;

		rb.isKinematic = false;
	}

	public void SetPosition(Vector3 pos)
	{
		rb.isKinematic = true;

		rb.position = pos;

		rb.isKinematic = false;
	}
				
	public float GetSteering()
	{
		return requestSteering;
	}

	public float GetThrottle()
	{
		return requestTorque;
	}

	public float GetBrake()
	{
		return requestBrake;
	}

	public Vector3 GetVelocity()
	{
		return rb.velocity;
	}

	public Vector3 GetAccelleration()
	{
		return acceleration;
	}

	public float GetOrient ()
	{
		Vector3 dir = transform.forward;
		return Mathf.Atan2( dir.z, dir.x);
	}

	public Transform GetTransform()
	{
		return this.transform;
	}

	public Vector3 GetPosition()
	{
		return this.transform.position;
	}
				
	public bool IsStill()
	{
		return rb.IsSleeping();
	}
		
	// Update is called once per frame
	void Update () {
	
		UpdateWheelPositions();
	}

	void FixedUpdate()
	{
		lastSteer = requestSteering;
		lastAccel = requestTorque;

		float throttle = requestTorque * maxTorque;
		float steerAngle = requestSteering * maxSteer;
        float brake = requestBrake;

		//front two tires.
		wheelColliders[2].steerAngle = steerAngle;
		wheelColliders[3].steerAngle = steerAngle;

		//four wheel drive at the moment
		foreach(WheelCollider wc in wheelColliders)
		{
			if(rb.velocity.magnitude < maxSpeed)
			{
				wc.motorTorque = throttle;
			}
			else
			{
				wc.motorTorque = 0.0f;
			}

			wc.brakeTorque = 400f * brake;
		}

		acceleration = rb.velocity - prevVelocity / Time.fixedDeltaTime;
	}

	void FlipUpright()
	{
		Quaternion rot = Quaternion.Euler(180f, 0f, 0f);
		this.transform.rotation = transform.rotation * rot;
		transform.position = transform.position + Vector3.up * 2;
	}

	void UpdateWheelPositions()
	{
		Quaternion rot;
		Vector3 pos;

		for(int i = 0; i < wheelColliders.Length; i++)
		{
			WheelCollider wc = wheelColliders[i];
			Transform tm = wheelMeshes[i];

			wc.GetWorldPose(out pos, out rot);

			tm.position = pos;
			tm.rotation = rot;
		}
	}
}
