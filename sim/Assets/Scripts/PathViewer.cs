using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PathViewer : MonoBehaviour {

	public CarPath path;
	public GameObject prefab;
	public LineRenderer line;
	public Transform startPos;
	Vector3 span = Vector3.zero;

	public float spanDist = 5f;

	public bool doLoadScriptPath = false;
	public bool doLoadPointPath = true;
	public bool doShowPath = false;

	public string pathToLoad = "none";

	void Awake () 	
	{
		InitNewRoad();			
	}

	public void InitNewRoad()
	{
		if(doLoadPointPath)
		{
			MakePointPath();
		}
		else if(doLoadScriptPath)
		{
			MakeScriptedPath();
		}
		if(doShowPath)
		{
			string dump = "";
			line.positionCount = path.nodes.Count;
			for(int iN = 0; iN < path.nodes.Count; iN++)
			{
				Vector3 np = path.nodes[iN].pos;
				line.SetPosition (iN, np);
				dump += string.Format("{0}\n", np);
			}
			Debug.Log(string.Format("Path dump:\n{0}", dump));
		}
	}

	public void DestroyRoad()
	{
		GameObject[] prev = GameObject.FindGameObjectsWithTag("pathNode");

		foreach(GameObject g in prev)
			Destroy(g);
	}

	public Vector3 GetPathStart()
	{
		return startPos.position;
	}

	public Vector3 GetPathEnd()
	{
		int iN = path.nodes.Count - 1;

		if(iN < 0)
			return GetPathStart();

		return path.nodes[iN].pos;
	}

	void MakePointPath()
	{
		TextAsset bindata = Resources.Load(pathToLoad) as TextAsset;

		if(bindata == null)
			return;

		string[] lines = bindata.text.Split('\n');

		path = new CarPath();

		Vector3 np = Vector3.zero;

		float offsetY = -0.1f;

		foreach(string line in lines)
		{
			string[] tokens = line.Split(',');

			if (tokens.Length != 3)
				continue;
			np.x = float.Parse(tokens[0]);
			np.y = float.Parse(tokens[1]) + offsetY;
			np.z = float.Parse(tokens[2]);
			PathNode p = new PathNode();
			p.pos = np;
			path.nodes.Add(p);
		}

	}

	void MakeScriptedPath()
	{
		TrackScript script = new TrackScript();

		if(script.Read(pathToLoad))
		{
			path = new CarPath();
			TrackParams tparams = new TrackParams();
			tparams.numToSet = 0;
			tparams.rotCur = Quaternion.identity;
			tparams.lastPos = startPos.position;

			float dY = 0.0f;
			float turn = 0f;

			Vector3 s = startPos.position;
			s.y = 0.5f;
			span.x = 0f;
			span.y = 0f;
			span.z = spanDist;
			float turnVal = 10.0f;

			foreach(TrackScriptElem se in script.track)
			{
				if(se.state == TrackParams.State.AngleDY)
				{
					turnVal = se.value;
				}
				else if(se.state == TrackParams.State.CurveY)
				{
					turn = 0.0f;
					dY = se.value * turnVal;
				}
				else
				{
					dY = 0.0f;
					turn = 0.0f;
				}

				for(int i = 0; i < se.numToSet; i++)
				{

					Vector3 np = s;
					PathNode p = new PathNode();
					p.pos = np;
					path.nodes.Add(p);

					turn = dY;

					Quaternion rot = Quaternion.Euler(0.0f, turn, 0f);
					span = rot * span.normalized;
					span *= spanDist;
					s = s + span;
				}

			}
		}
	}
}