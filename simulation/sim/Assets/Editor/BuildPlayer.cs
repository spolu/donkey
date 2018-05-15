using UnityEditor;
using UnityEngine;

public class BuildPlayer
{
	static void PerformBuildLinux64 ()
	{
		BuildPlayerOptions buildPlayerOptions = new BuildPlayerOptions();
		buildPlayerOptions.scenes = new[] { "Assets/Scenes/warehouse.unity" };
		buildPlayerOptions.locationPathName = "../build/sim";
		buildPlayerOptions.target = BuildTarget.StandaloneLinux64;
		buildPlayerOptions.options = BuildOptions.None;
		BuildPipeline.BuildPlayer(buildPlayerOptions);
	}

	static void PerformBuildOSX ()
	{
		BuildPlayerOptions buildPlayerOptions = new BuildPlayerOptions();
		buildPlayerOptions.scenes = new[] { "Assets/Scenes/warehouse.unity" };
		buildPlayerOptions.locationPathName = "../build/sim";
		buildPlayerOptions.target = BuildTarget.StandaloneOSX;
		buildPlayerOptions.options = BuildOptions.None;
		BuildPipeline.BuildPlayer(buildPlayerOptions);
	}
}
