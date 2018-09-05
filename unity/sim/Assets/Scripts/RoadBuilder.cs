using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RoadBuilder : MonoBehaviour {
    
	public float roadOffsetW = 0.0f;

	public GameObject roadPrefabMesh;

	public Texture2D[] roadTextures;

	Texture2D customRoadTexure;

	GameObject createdRoad;

	void Awake ()
	{
	}

	void Start()
	{
	}

	public void DestroyRoad()
	{
		GameObject[] prev = GameObject.FindGameObjectsWithTag("road_mesh");

		foreach(GameObject g in prev)
			Destroy(g);
	}

	public void SetRoadVariation(int iVariation)
	{
		if(roadTextures.Length > 0)
			customRoadTexure = roadTextures[ iVariation % roadTextures.Length ];
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
		if(pathData == null)
			return null;

		string[] lines = pathData.Split(';');

		// Debug.Log(string.Format("found {0} path points to load", lines.Length));

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

	public CarPath BuildRoad(string pathData, float roadWidth, int roadTexture, float roadTextureLength)
	{
		//roadTextureLength is the distance in meters of the texture pattern in reality
		CarPath path = MakePointPath(pathData);

		if (path != null)
			InitRoad(path, roadWidth, roadTexture, roadTextureLength);

		return path;
	}

	public void InitRoad(CarPath path, float roadWidth, int roadTexture, float roadTextureLength)
	{
		SetRoadVariation (roadTexture % roadTextures.Length);

		GameObject go = GameObject.Instantiate(roadPrefabMesh);
		MeshRenderer mr = go.GetComponent<MeshRenderer>();
		MeshFilter mf = go.GetComponent<MeshFilter>();
		Mesh mesh = new Mesh();

		mf.mesh = mesh;
		createdRoad = go;

		mr.material.mainTexture = customRoadTexure;

		go.tag = "road_mesh";

		int numQuads = path.nodes.Count - 1;
		int numVerts = (numQuads + 1) * 2;
		int numTris = numQuads * 2;

		Vector3[] vertices = new Vector3[numVerts];

		int numTriIndexes = numTris * 3;
		int[] tri = new int[numTriIndexes];

		int numNormals = numVerts;
		Vector3[] normals = new Vector3[numNormals];

		int numUvs = numVerts;
		Vector2[] uv = new Vector2[numUvs];

		for(int iN = 0; iN < numNormals; iN++)
			normals[iN] = Vector3.up;

		int iNode = 0;

		Vector3 posA = Vector3.zero;
		Vector3 posB = Vector3.zero;

		Vector3 lastLeftPos = Vector3.zero;
		Vector3 lastRightPos = Vector3.zero;

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

			}
			else
			{
				PathNode nodeA = path.nodes[iNode];
				PathNode nodeB = path.nodes[0];
				posA = nodeA.pos;
				posB = nodeB.pos;
			}

			vLength = posB - posA;
			vWidth = Vector3.Cross(vLength, Vector3.up);

			Vector3 leftPos = posA + vWidth.normalized * roadWidth + Vector3.up * roadOffsetW;
			Vector3 rightPos = posA - vWidth.normalized * roadWidth + Vector3.up * roadOffsetW;

			// Prevent visual artifact in tight turns
			if (lastLeftPos != Vector3.zero) {
				if (Vector3.Dot (vLength, leftPos - lastLeftPos) < 0.0) {
					leftPos = lastLeftPos;
				}
			}
			if (lastRightPos != Vector3.zero) {
				if (Vector3.Dot (vLength, rightPos - lastRightPos) < 0.0) {
					rightPos = lastRightPos;
				}
			}
			lastLeftPos = leftPos;
			lastRightPos = rightPos;

			vertices [iVert] = leftPos;
			vertices [iVert + 1] = rightPos;

			uv[iVert] = new Vector2(vLength.magnitude / roadTextureLength * iNode, 0.0f);
			uv[iVert + 1] = new Vector2(vLength.magnitude / roadTextureLength * iNode, 1.0f);

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
