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
	public CameraSensor camSensor;
	public RoadBuilder roadBuilder;

	private ICar car;
	private SocketIOComponent _socket;
	private bool connected = false;
	private int clientID = 0;

	private float timeScale = 1.0f;
	private float stepInterval = 0.30f;
	private int captureFrameRate = 0;

	private float lastResume = 0.0f;
	private float lastTelemetry = 0.0f;
	private float lastPause = 0.0f;

	private float fpsInterval = 3.0f;
	private float fpsAccumulator = 0.0f;
	private int fpsFrameCount  = 0;
	private float fpsValue = 0.0f;
	private string socketIOUrl = "ws://127.0.0.1:9999/socket.io/?EIO=4&transport=websocket";

	private float lastRequestedSteering = 0.0f;
	private float lastRequestedThrottle = 0.0f;
	private float lastRequestedBrake = 0.0f;

	private bool doPause = true;

	private float carStartY = 0.001f;

    private Vector3 prevVelocity = Vector3.zero;

	void Awake()
	{
		string[] args = System.Environment.GetCommandLineArgs ();

		for (int i = 0; i < args.Length - 1; i++) {
			if (args [i] == "-simulationClientID") {
				clientID = int.Parse(args[i+1]);
			}
			if (args [i] == "-simulationTimeScale") {
				timeScale = float.Parse(args[i+1]);
			}
			if (args [i] == "-simulationStepInterval") {
				stepInterval = float.Parse(args[i+1]);
			}
			if (args [i] == "-simulationCaptureFrameRate") {
				captureFrameRate = int.Parse(args[i+1]);
			}
			if (args [i] == "-socketIOPort") {
				socketIOUrl = "ws://127.0.0.1:" + args[i+1] + "/socket.io/?EIO=4&transport=websocket";
			}
		}

		_socket = GetComponent<SocketIOComponent>();
		_socket.url = socketIOUrl;
	}

	void Start()
	{
		// Debug.Log ("SimulationController initializing");

		Time.captureFramerate = captureFrameRate;

		_socket.On ("open", OnOpen);
		_socket.On ("step", OnStep);
		_socket.On ("exit", OnExit);
		_socket.On ("reset", OnReset);

		car = carObject.GetComponent<ICar>();
	}

	private void OnEnable()
	{
		// Debug.Log("SimulationController enabling");
	}

	private void OnDisable()
	{
		// Debug.Log ("SimulationController disabling");

		car.RequestThrottle (0.0f);
		car.RequestBrake (1.0f);
	}
		
	public void Send(SimMessage m)
	{
		// Debug.Log ("Direct sending: type=" + m.type);

		m.json.AddField ("id", clientID);
		_socket.Emit (m.type, m.json);
	}

	// TELEMETRY / UPDATE / TIMESCALE

	void Pause()
	{
		// Debug.Log ("Pause: time=" + Time.time);
		lastPause = Time.time;
		if (doPause) {
			Time.timeScale = 0.0f;
		}
	}
	void Resume()
	{
		// Debug.Log ("Resume: time=" + Time.time);
		lastResume = Time.time;
		Time.timeScale = timeScale;
	}

	private void FixedUpdate()
	{
		// Debug.Log ("FIXED time=" + Time.time + " position=" + car.GetPosition ().z);
	}

	private void Update()
	{
		// Debug.Log ("UDPATE time=" + Time.time + " position=" + car.GetPosition ().z);

		if (connected) {
			if (Time.time >= lastResume + stepInterval && Time.time > lastTelemetry) {
				Vector3 acc = (car.GetVelocity() - prevVelocity) / (Time.time - lastTelemetry);
				prevVelocity = car.GetVelocity();
				lastTelemetry = Time.time;

				// Debug.Log ("Sending Telemetry: connected=" + connected +
				//    " time=" + Time.time + " last_resume=" + lastResume + " last_pause=" + lastPause);

				SimMessage m = new SimMessage ();
				m.json = new JSONObject (JSONObject.Type.OBJECT);

				m.type = "telemetry";

				m.json.AddField ("time", Time.time);
				m.json.AddField ("last_resume", lastResume);
				m.json.AddField ("last_pause", lastPause);
				m.json.AddField ("last_telemetry", lastTelemetry);
				m.json.AddField ("time_scale", Time.timeScale);
				m.json.AddField ("fps", fpsValue);
				m.json.AddField ("delta", Time.deltaTime);
				m.json.AddField ("fixed_delta", Time.fixedDeltaTime);

				m.json.AddField ("steering", car.GetSteering ());
				m.json.AddField ("throttle", car.GetThrottle ());
				m.json.AddField ("brake", car.GetBrake ());

				m.json.AddField ("camera", System.Convert.ToBase64String (camSensor.GetImage ().EncodeToJPG ()));

				JSONObject position = new JSONObject (JSONObject.Type.OBJECT);
				position.AddField ("x", car.GetPosition ().x);
				position.AddField ("y", car.GetPosition ().y);
				position.AddField ("z", car.GetPosition ().z);
				m.json.AddField ("position", position);

				JSONObject velocity = new JSONObject (JSONObject.Type.OBJECT);
				velocity.AddField ("x", car.GetVelocity ().x);
				velocity.AddField ("y", car.GetVelocity ().y);
				velocity.AddField ("z", car.GetVelocity ().z);
				m.json.AddField ("velocity", velocity);
                
				JSONObject acceleration = new JSONObject (JSONObject.Type.OBJECT);
				acceleration.AddField ("x", acc.x);
				acceleration.AddField ("y", acc.y);
				acceleration.AddField ("z", acc.z);
				m.json.AddField ("acceleration", acceleration);

				JSONObject angularVelocity = new JSONObject(JSONObject.Type.OBJECT);
				angularVelocity.AddField("x", car.GetAngularVelocity().x);
				angularVelocity.AddField("y", car.GetAngularVelocity().y);
				angularVelocity.AddField("z", car.GetAngularVelocity().z);
				m.json.AddField("angular_velocity", angularVelocity);

				Send (m);
				Pause ();
			}
		}

		fpsAccumulator += Time.deltaTime;
		++fpsFrameCount;

		if( fpsAccumulator > fpsInterval)
		{
			fpsValue = fpsFrameCount / fpsAccumulator;
			fpsAccumulator = 0.0f;
			fpsFrameCount = 0;			
		}

	}

	// SOCKET IO HANDLERS

	void OnOpen(SocketIOEvent ev)
	{
		if (_socket.sid == null) {
			return;
		}
			
		// Debug.Log ("Received: type=open sid=" + _socket.sid);

		SimMessage m = new SimMessage ();
		m.json = new JSONObject (JSONObject.Type.OBJECT);
		m.type = "hello";

		m.json.AddField ("client_id", clientID);	
		m.json.AddField ("time_scale", timeScale);	
		m.json.AddField ("step_interval", stepInterval);	

		Send (m);
	}

	void OnReset(SocketIOEvent ev)
	{		
		// Debug.Log ("Received: type=reset sid=" + _socket.sid);

		connected = true;
		lastPause = Time.time + 999.0f;
		lastResume = Time.time;
		lastTelemetry = 0.0f;

		// Parse track
		string trackPath = ev.data.GetField("track_path").str;
		float trackWidth = float.Parse(ev.data.GetField("track_width").str);
		string startPosition = ev.data.GetField("start_position").str;
		float startAngle = float.Parse(ev.data.GetField("start_angle").str);

		// Redraw the track
		roadBuilder.DestroyRoad();
		CarPath path = roadBuilder.BuildRoad(trackPath, trackWidth);

		Vector3 trackStartPos = Vector3.zero;
        // if (path != null):
        //    trackStartPos = path.nodes[0].pos;
		if (startPosition.Length > 0)
		{
			string[] tokens = startPosition.Split(',');
			if (tokens.Length == 3)
			{
				trackStartPos.x = float.Parse(tokens[0]);
				trackStartPos.z = float.Parse(tokens[2]);
			}
		}
		trackStartPos.y = carStartY;

		Debug.Log("Received: type=open sid=" + _socket.sid);

		car.Set(trackStartPos, Quaternion.AngleAxis(Mathf.Rad2Deg * startAngle, Vector3.up));

		Resume ();
	}

	void OnStep(SocketIOEvent ev)
	{
		// Debug.Log ("Received: type=step sid=" + _socket.sid + " data=" + ev.data);

        // Apply controls from last step to simulate the delay incurred by running our predictions.

		car.RequestSteering (lastRequestedSteering);
		car.RequestThrottle (lastRequestedThrottle);
		car.RequestBrake (lastRequestedBrake);

		lastRequestedThrottle = float.Parse(ev.data.GetField("throttle").str);
		lastRequestedSteering = float.Parse(ev.data.GetField("steering").str);
		lastRequestedBrake = float.Parse(ev.data.GetField("brake").str);
        
		Resume ();
	}

	void OnExit(SocketIOEvent ev)
	{
		// Debug.Log ("Received: type=exit sid=" + _socket.sid);

		Application.Quit ();
	}
}