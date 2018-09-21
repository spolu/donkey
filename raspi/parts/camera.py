import os
import time
import cv2
import glob

import numpy as np

class PiCamera:
    def __init__(self, resolution=(120, 160), framerate=20):
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        resolution = (resolution[1], resolution[0])
        # initialize the camera and stream
        self.camera = PiCamera() #PiCamera gets resolution (height, width)
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="rgb", use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.on = True

        print('PiCamera loaded... warming camera')
        time.sleep(2)


    def run_threaded(self):
        return self.raw, self.camera

    def run(self):
        f = next(self.stream)
        frame = f.array
        self.rawCapture.truncate(0)

        return self.output_from_frame(frame)

    def output_from_frame(self, frame):
        # get b,g,r as cv2 uses BGR and not RGB for colors
        b,g,r = cv2.split(frame)
        rgb_img = cv2.merge([r,g,b])

        raw = cv2.imencode(".jpg", rgb_img)[1].tostring()

        camera = cv2.imdecode(
            np.fromstring(camera_raw, np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )

        return raw, camera

    def update(self):
        print("PiCamera update")
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)
            self.camera, self.raw = self.output_from_frame(self.frame)

            # if the thread indicator variable is set, stop the thread
            if not self.on:
                break

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        print('stoping PiCamera')
        time.sleep(.5)
        self.stream.close()
        self.rawCapture.close()
        self.camera.close()

