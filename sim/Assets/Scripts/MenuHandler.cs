using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MenuHandler : MonoBehaviour {

	public GameObject Logger;
	public GameObject NetworkSteering;
	public GameObject menuPanel;
	public GameObject stopPanel;
	public GameObject exitPanel;

    public void Awake()
    {
        //keep it processing even when not in focus.
        Application.runInBackground = true;

        //Set desired frame rate as high as possible.
        Application.targetFrameRate = 60;

		menuPanel.SetActive(true);
        stopPanel.SetActive(false);
        exitPanel.SetActive(true);
    }

	public void OnManualGenerateTrainingData()
	{
		Logger.SetActive(true);
        
		menuPanel.SetActive(false);
        stopPanel.SetActive(true);
        exitPanel.SetActive(false);
    }

	public void OnUseNNNetworkSteering()
	{
		NetworkSteering.SetActive(true);
		menuPanel.SetActive(false);
        stopPanel.SetActive(true);
        exitPanel.SetActive(false);
    }

	public void OnManualDrive()
	{
		menuPanel.SetActive(false);
        stopPanel.SetActive(true);
        exitPanel.SetActive(false);
    }

    public void OnStop()
    {
        Logger.SetActive(false);
        NetworkSteering.SetActive(false);


        menuPanel.SetActive(true);
        stopPanel.SetActive(false);
        exitPanel.SetActive(true);
    }

}
