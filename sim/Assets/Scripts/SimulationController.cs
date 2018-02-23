using SocketIO;
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

	private float stepInterval = 0.10f;
	private float lastResume = 0.0f;
	private float lastTelemetry = 0.0f;
	private float lastPause = 0.0f;


	void Start()
	{
		Debug.Log ("SimulationController initializing");

``
		_socket = GetComponent<SocketIOComponent>();

		_socket.On ("open", OnOpen);
		_socket.On ("step", OnStep);
		_socket.On ("exit", OnExit);
		_socket.On ("reset", OnReset);

		car = carObject.GetComponent<ICar>();

		car.SavePosRot ();
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

		m.json.AddField ("id", clientID);
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
		if (connected) {
			if (Time.time >= lastResume + stepInterval && Time.time > lastTelemetry) {
				lastTelemetry = Time.time;
				Debug.Log ("Sending Telemetry: connected=" + connected + 
					" time=" + Time.time + " last_resume="+ lastResume + " last_pause=" + lastPause);

				SimMessage m = new SimMessage();
				m.json = new JSONObject(JSONObject.Type.OBJECT);

				m.type = "telemetry";

				m.json.AddField ("time", Time.time);	

				m.json.AddField ("steering", car.GetSteering());
				m.json.AddField ("throttle", car.GetThrottle());
				m.json.AddField ("brake", car.GetHandBrake());

				m.json.AddField ("camera", System.Convert.ToBase64String(CameraHelper.CaptureFrame(camSensor)));

				JSONObject position = new JSONObject(JSONObject.Type.OBJECT);
				position.AddField ("x", car.GetPosition().x);
				position.AddField ("y", car.GetPosition().y);
				position.AddField ("z", car.GetPosition().z);
				m.json.AddField ("position", position);

				JSONObject velocity = new JSONObject(JSONObject.Type.OBJECT);
				velocity.AddField ("x", car.GetVelocity().x);
				velocity.AddField ("y", car.GetVelocity().y);
				velocity.AddField ("z", car.GetVelocity().z);
				m.json.AddField ("velocity", velocity);

				JSONObject acceleration = new JSONObject(JSONObject.Type.OBJECT);
				acceleration.AddField ("x", car.GetAccelleration().x);
				acceleration.AddField ("y", car.GetAccelleration().y);
				acceleration.AddField ("z", car.GetAccelleration().z);
				m.json.AddField ("acceleration", acceleration);

				Send (m);
				Pause ();
			}
		}
	}

	// SOCKET IO HANDLERS

	void OnOpen(SocketIOEvent ev)
	{
		if (_socket.sid == null) {
			return;
		}
			
		Debug.Log ("Received: type=open sid=" + _socket.sid);

		SimMessage m = new SimMessage ();
		m.json = new JSONObject (JSONObject.Type.OBJECT);
		m.type = "hello";

		Send (m);
	}

	void OnReset(SocketIOEvent ev)
	{		
		Debug.Log ("Received: type=reset sid=" + _socket.sid);

		connected = true;
		lastPause = Time.time + 999.0f;
		lastResume = Time.time;
		lastTelemetry = 0.0f;
		Time.timeScale = 1.0f;

		// Reset the car to its initial state.
		car.RestorePosRot ();
	}

	void OnStep(SocketIOEvent ev)
	{
		Debug.Log ("Received: type=step sid=" + _socket.sid + " data=" + ev.data);
						
		float steeringReq = float.Parse(ev.data.GetField("steering").str);
		float throttleReq = float.Parse(ev.data.GetField("throttle").str);
		float brakeReq = float.Parse(ev.data.GetField("brake").str);

		car.RequestSteering (steeringReq);
		car.RequestThrottle (throttleReq);
		car.RequestFootBrake (brakeReq);
		car.RequestHandBrake (0.0f);

		Resume ();
	}

	void OnExit(SocketIOEvent ev)
	{
		Debug.Log ("Received: type=exit sid=" + _socket.sid);
		Application.Quit ();
	}
}