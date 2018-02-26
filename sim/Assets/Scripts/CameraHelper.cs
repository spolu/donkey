using UnityEngine;

public static class CameraHelper
{
	public static byte[] CaptureFrame(Camera camera)
	{
		var targetTexture = new RenderTexture(camera.pixelWidth, camera.pixelHeight, 24);
		camera.targetTexture = targetTexture;
		camera.Render();
		camera.targetTexture = null;

		RenderTexture.active = targetTexture;
		Texture2D texture2D = new Texture2D(targetTexture.width, targetTexture.height, TextureFormat.RGB24, false);
		texture2D.ReadPixels(new Rect(0, 0, targetTexture.width, targetTexture.height), 0, 0);
    	texture2D.Apply();
		RenderTexture.active = null;

   		byte[] image = texture2D.EncodeToJPG();
    	//byte[] image = texture2D.EncodeToPNG();
    	//byte[] image = texture2D.GetRawTextureData();

		// Required to prevent leaking the texture
    	Object.DestroyImmediate(texture2D); 
		Object.DestroyImmediate(targetTexture);

    	return image;
  	}
}
