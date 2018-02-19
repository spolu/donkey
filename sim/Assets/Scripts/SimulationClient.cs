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
public class SimulationClient : MonoBehaviour
{

	public GameObject carObject;
	public Camera camSensor;

	private ICar car;
	private List<SimMessage> messages;
	private SocketIOComponent _socket;
	private bool connected = false;

	private float stepInterval = 0.10f;
	private float lastResume = 0.0f;
	private float lastPause = 999.0f;


	void Start()
	{
		Init();
	}

	private void OnEnable()
	{
		Debug.Log("SimulationClient enabling");
	}

	private void OnDisable()
	{
		Debug.Log ("SimulationClient disabling");

		car.RequestFootBrake(1.0f);
	}

	private void Init()
	{
		if (messages != null)
			return;

		Debug.Log ("SimulationClient initializing");

		_socket = GetComponent<SocketIOComponent>();
		_socket.On ("open", OnOpen);
		_socket.On ("step", OnStep);
		_socket.On ("exit", OnExit);
		// _socket.On("reset", OnReset);

		messages = new List<SimMessage>();

		car = carObject.GetComponent<ICar>();
	}

	public void Send(SimMessage m)
	{
		Debug.Log ("Direct sending: type=" + m.type);
		_socket.Emit (m.type, m.json);
		// lock (this)
		// {
		// 	messages.Add (m);
		// }
	}

	// TELEMETRY / UPDATE / TIMESCALE

	void Pause()
	{
		lastPause = Time.time;
		Time.timeScale = 0;
	}
	void Resume()
	{
		lastResume = Time.time;
		Time.timeScale = 1;
	}

	private void Update()
	{
		if (connected)
		{
			if (Time.time >= lastResume + stepInterval && Time.time < lastPause) 
			{
				Debug.Log ("Sending Telemetry: connected=" + connected + 
					" time=" + Time.time + " last_resume="+ lastResume + " last_pause=" + lastPause);

				SimMessage m = new SimMessage();
				m.json = new JSONObject(JSONObject.Type.OBJECT);

				m.type = "telemetry";

				m.json.AddField ("steering_angle", car.GetSteering());
				m.json.AddField ("throttle", car.GetThrottle());
				m.json.AddField ("speed", car.GetVelocity().magnitude);
				m.json.AddField ("camera", System.Convert.ToBase64String(CameraHelper.CaptureFrame(camSensor)));

				Send (m);
				Pause ();
			}
		}
	}

	// SOCKET IO HANDLERS

	void OnOpen(SocketIOEvent ev)
	{
		Debug.Log ("Received: type=open");
		connected = true;
	}

	void OnStep(SocketIOEvent ev)
	{
		Debug.Log ("Received: type=step sid=" + _socket.sid);
		JSONObject jsonObject = ev.data;

		float steeringReq = float.Parse(jsonObject.GetField("steering_angle").str);
		float throttleReq = float.Parse(jsonObject.GetField("throttle").str);
		float breakReq = float.Parse(jsonObject.GetField("break").str);

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