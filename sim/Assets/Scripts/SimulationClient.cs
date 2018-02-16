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
	private Thread sendThread;
	private List<SimMessage> messages;
	private SocketIOComponent _socket;
	private bool connected = false;

	private float stepInterval = 0.10f;
	private float lastResume = 0.0f;


	void Start()
	{
		Init();
	}

	private void OnEnable()
	{
		Debug.Log("SimulationClient SendThread starting");

		sendThread = new Thread(SendThread);
		sendThread.Start();
	}

	private void OnDisable()
	{
		car.RequestFootBrake(1.0f);
		sendThread.Abort();
	}

	private void Init()
	{
		Debug.Log("SimulationClient initializing");

		if (messages != null)
			return;

		_socket = GetComponent<SocketIOComponent>();
		_socket.On("open", OnOpen);
		_socket.On("step", OnStep);
		_socket.On("exit", OnExit);
		// _socket.On("reset", OnReset);

		messages = new List<SimMessage>();

		car = carObject.GetComponent<ICar>();
	}

	// SEND THREAD

	public void SendThread()
	{
		lock (messages)
		{
			if(messages.Count != 0 && connected)
			{
				foreach(SimMessage m in messages)
				{
					Debug.Log("SendThread sending: type=" + m.type);

					_socket.Emit(m.type, m.json);
				}

				messages.Clear();
			}
		}
	}

	public void Send(SimMessage m)
	{
		lock (messages)
		{
			messages.Add(m);
		}
	}

	// TELEMETRY / UPDATE / TIMESCALE

	void Pause()
	{
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
			if (Time.time >= lastResume + stepInterval) {

				SimMessage m = new SimMessage();
				m.json = new JSONObject(JSONObject.Type.OBJECT);

				m.type = "telemetry";

				m.json.AddField("steering_angle", car.GetSteering());
				m.json.AddField("throttle", car.GetThrottle());
				m.json.AddField("speed", car.GetVelocity().magnitude);
				m.json.AddField("camera", System.Convert.ToBase64String(CameraHelper.CaptureFrame(camSensor)));

				Send(m);
				Pause();
			}
		}
	}

	// SOCKET IO HANDLERS

	void OnOpen(SocketIOEvent obj)
	{
		Debug.Log("Received: type=open");
		connected = true;
	}

	void OnStep(SocketIOEvent obj)
	{
		Debug.Log("Received: type=step");
		JSONObject jsonObject = obj.data;

		float steeringReq = float.Parse(jsonObject.GetField("steering_angle").str);
		float throttleReq = float.Parse(jsonObject.GetField("throttle").str);
		float breakReq = float.Parse(jsonObject.GetField("break").str);

		car.RequestSteering(steeringReq);
		car.RequestThrottle(throttleReq);
		car.RequestFootBrake(breakReq);
		car.RequestHandBrake(0.0f);

		Resume();
	}

	void OnExit(SocketIOEvent obj)
	{
		Debug.Log("Received: type=exit");
		Application.Quit();
	}
}