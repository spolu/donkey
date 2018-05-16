using UnityEngine;
using System.Collections;

public class CameraController : MonoBehaviour {

    public GameObject donkey;
    private Vector3 offset;
    
    void Start () 
    {
        offset = transform.position - donkey.transform.position;
    }
    
    void LateUpdate () 
    {
        transform.position = donkey.transform.position + offset;
    }
}