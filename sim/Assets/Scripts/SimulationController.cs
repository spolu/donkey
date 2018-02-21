﻿using SocketIO;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

public class SimMessage
{
	public JSONObject json;
	public string type;
}

[RequireComponent (typeof(SocketIOComponent))]
public class SimulationController : MonoBehaviour
{

	public GameObject carObject;
	public Camera camSensor;

	private ICar car;
	private SocketIOComponent _socket;
	private bool connected = false;
	private int clientID = 0;

	private float stepInterval = 0.05f;
	private float lastResume = 0.0f;
	private float lastTelemetry = 0.0f;
	private float lastPause = 0.0f;


	void Start()
	{
		Debug.Log ("SimulationController initializing");

		string[] args = System.Environment.GetCommandLineArgs ();
		if (args.Length > 0) {
			Debug.Log ("Arguments: args=" + args);
		}

		_socket = GetComponent<SocketIOComponent>();

		_socket.On ("open", OnOpen);
		_socket.On ("step", OnStep);
		_socket.On ("exit", OnExit);
		_socket.On ("reset", OnReset);

		car = carObject.GetComponent<ICar>();
	}

	private void OnEnable()
	{
		Debug.Log("SimulationController enabling");
	}

	private void OnDisable()
	{
		Debug.Log ("SimulationController disabling");

		car.RequestFootBrake(1.0f);
	}
		
	public void Send(SimMessage m)
	{
		Debug.Log ("Direct sending: type=" + m.type);
		_socket.Emit (m.type, m.json);
	}

	// TELEMETRY / UPDATE / TIMESCALE

	void Pause()
	{
		Debug.Log ("Pause: time=" + Time.time);
		lastPause = Time.time;
		Time.timeScale = 0.0f;
	}
	void Resume()
	{
		Debug.Log ("Resume: time=" + Time.time);
		lastResume = Time.time;
		Time.timeScale = 1.0f;
	}

	private void Update()
	{
		if (connected)
		{
			if (Time.time >= lastResume + stepInterval && Time.time > lastTelemetry) 
			{
				lastTelemetry = Time.time;
				Debug.Log ("Sending Telemetry: connected=" + connected + 
					" time=" + Time.time + " last_resume="+ lastResume + " last_pause=" + lastPause);

				SimMessage m = new SimMessage();
				m.json = new JSONObject(JSONObject.Type.OBJECT);

				m.type = "telemetry";

				m.json.AddField ("time", Time.time);				
				m.json.AddField ("steering", car.GetSteering());
				m.json.AddField ("throttle", car.GetThrottle());
				m.json.AddField ("speed", car.GetVelocity().magnitude);
				m.json.AddField ("camera", System.Convert.ToBase64String(CameraHelper.CaptureFrame(camSensor)));
				JSONObject position = new JSONObject(JSONObject.Type.OBJECT);
				position.AddField ("x", car.GetPosition().x);
				position.AddField ("y", car.GetPosition().y);
				position.AddField ("z", car.GetPosition().z);
				m.json.AddField ("position", position);

				Send (m);
				Pause ();
			}
		}
	}

	// SOCKET IO HANDLERS

	void OnOpen(SocketIOEvent ev)
	{
		Debug.Log ("Received: type=open");

		SimMessage m = new SimMessage();
		m.json = new JSONObject(JSONObject.Type.OBJECT);
		m.type = "hello";
		m.json.AddField ("id", clientID);
		Send (m);
	}

	void OnReset(SocketIOEvent ev)
	{		
		Debug.Log ("Received: type=reset sid=" + _socket.sid);

		// TODO: reset simulation to initial state?

		connected = true;
		lastPause = Time.time + 999.0f;
		lastResume = Time.time;
		lastTelemetry = 0.0f;
		Time.timeScale = 1.0f;
	}

	void OnStep(SocketIOEvent ev)
	{
		Debug.Log ("Received: type=step sid=" + _socket.sid + " step=" + ev.data);
						
		float steeringReq = float.Parse(ev.data.GetField("steering").str);
		float throttleReq = float.Parse(ev.data.GetField("throttle").str);
		float breakReq = float.Parse(ev.data.GetField("break").str);

		car.RequestSteering (steeringReq);
		car.RequestThrottle (throttleReq);
		car.RequestFootBrake (breakReq);
		car.RequestHandBrake (0.0f);

		Resume ();
	}

	void OnExit(SocketIOEvent ev)
	{
		Debug.Log ("Received: type=exit sid=" + _socket.sid);
		Application.Quit ();
	}
}