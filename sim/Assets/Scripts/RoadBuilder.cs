using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RoadBuilder : MonoBehaviour {

	public float roadWidth = 1.0f;
	public float roadHeightOffset = 0.0f;
	public float roadOffsetW = 0.0f;
	public string pathToLoad = "none";
		
	public GameObject roadPrefabMesh;

	public int iRoadTexture = 0;
	public Texture2D[] roadTextures;
	public float[] roadOffsets;
	public float[] roadWidths;

	Texture2D customRoadTexure;

	GameObject createdRoad;

	void Awake () 	
	{
		BuildRoad(null);
	}

	void Start()
	{
	}

	public void DestroyRoad()
	{
		GameObject[] prev = GameObject.FindGameObjectsWithTag("road_mesh");

		foreach(GameObject g in prev)
			Destroy(g);

		//advance road index into texture list.
		iRoadTexture += 1;
	}

	public void SetNewRoadVariation(int iVariation)
	{
		if(roadTextures.Length > 0)		
			customRoadTexure = roadTextures[ iVariation % roadTextures.Length ];

		if(roadOffsets.Length > 0)
			roadOffsetW = roadOffsets[ iVariation % roadOffsets.Length ];

		if(roadWidths.Length > 0)
			roadWidth = roadWidths[ iVariation % roadWidths.Length ];
		
	}

	public void NegateYTiling()
	{
		//todo
		if(createdRoad == null)
			return;
		
		MeshRenderer mr = createdRoad.GetComponent<MeshRenderer>();
		Vector2 ms = mr.material.mainTextureScale;
		ms.y *= -1.0f;
		mr.material.mainTextureScale = ms;
	}

	CarPath MakePointPath(string pathData)
	{
		// string pData = pathData;

		// if(pathData == null) {
		//	//load default path as default
		//	TextAsset bindata = Resources.Load(pathToLoad) as TextAsset;
		//	pData = bindata.text;
		// }

		if(pathData == null)
			return null;

		string[] lines = pathData.Split(';');

		Debug.Log(string.Format("found {0} path points. to load", lines.Length));

		CarPath path = new CarPath();

		Vector3 np = Vector3.zero;

		float offsetY = 0.0f;

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

		return path;
	}

	public CarPath BuildRoad(string pathData)
	{
		CarPath path = MakePointPath(pathData);

		if (path != null)
			InitRoad(path);
		
		return path;
	}


	public void InitRoad(CarPath path)
	{
		
		GameObject go = GameObject.Instantiate(roadPrefabMesh);
		MeshRenderer mr = go.GetComponent<MeshRenderer>();
		MeshFilter mf = go.GetComponent<MeshFilter>();
		Mesh mesh = new Mesh();
		mf.mesh = mesh;
		createdRoad = go;

		if(customRoadTexure != null)
		{
			mr.material.mainTexture = customRoadTexure;
		}
		else if(roadTextures != null && iRoadTexture < roadTextures.Length)
		{
			Texture2D t = roadTextures[iRoadTexture];

			if(mr != null && t != null)
			{
				mr.material.mainTexture = t;
			}
		}

		go.tag = "road_mesh";

		int numQuads = path.nodes.Count - 1;
		int numVerts = (numQuads + 1) * 2;
		int numTris = numQuads * 2;

		Vector3[] vertices = new Vector3[numVerts];

		int numTriIndecies = numTris * 3;
		int[] tri = new int[numTriIndecies];

		int numNormals = numVerts;
		Vector3[] normals = new Vector3[numNormals];

		int numUvs = numVerts;
		Vector2[] uv = new Vector2[numUvs];

		for(int iN = 0; iN < numNormals; iN++)
			normals[iN] = Vector3.up;

		int iNode = 0;

		Vector3 posA = Vector3.zero;
		Vector3 posB = Vector3.zero;

		Vector3 vLength = Vector3.one;
		Vector3 vWidth = Vector3.one;

		for(int iVert = 0; iVert < numVerts; iVert += 2)
		{
			if(iNode + 1 < path.nodes.Count)
			{
				PathNode nodeA = path.nodes[iNode];
				PathNode nodeB = path.nodes[iNode + 1];
				posA = nodeA.pos;
				posB = nodeB.pos;

				vLength = posB - posA;
				vWidth = Vector3.Cross(vLength, Vector3.up);

				posA.y += roadHeightOffset;
			}
			else
			{
				PathNode nodeA = path.nodes[iNode];
				posA = nodeA.pos;
				posA.y += roadHeightOffset;
			}

			Vector3 leftPos = posA + vWidth.normalized * roadWidth + vWidth.normalized * roadOffsetW;
			Vector3 rightPos = posA - vWidth.normalized * roadWidth + vWidth.normalized * roadOffsetW;

			vertices[iVert] = leftPos;
			vertices[iVert + 1] = rightPos;

			uv[iVert] = new Vector2(0.2f * iNode, 0.0f);
			uv[iVert + 1] = new Vector2(0.2f * iNode, 1.0f);

			iNode++;
		}

		int iVertOffset = 0;
		int iTriOffset = 0;

		for(int iQuad = 0; iQuad < numQuads; iQuad++)
		{
			tri[0 + iTriOffset] = 0 + iVertOffset;
			tri[1 + iTriOffset] = 2 + iVertOffset;
			tri[2 + iTriOffset] = 1 + iVertOffset;

			tri[3 + iTriOffset] = 2 + iVertOffset;
			tri[4 + iTriOffset] = 3 + iVertOffset;
			tri[5 + iTriOffset] = 1 + iVertOffset;

			iVertOffset += 2;
			iTriOffset += 6;
		}


		mesh.vertices = vertices;
		mesh.triangles = tri;
		mesh.normals = normals;
		mesh.uv = uv;

		mesh.RecalculateBounds();
	}
}
