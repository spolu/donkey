import os
import time
import numpy as np
import glob
import sys
import argparse
import cv2

 # import pdb; pdb.set_trace()

class JetsonCamera:
    def __init__(self, resolution=(1280, 720), framerate=120):
        resolution = (resolution[1], resolution[0])
        framerate = framerate
        dev = 1
        # initialize the camera and stream
        # usb board camera
        gst_str = ('v4l2src device=/dev/video{} ! '
                        'video/x-raw, width=640, height=480 !'
                        'videoscale ! '
                        'video/x-raw, width=(int){}, height=(int){} !'
                        'videoconvert !'
                        'video/x-raw, format=RGBA !'
                        'videoconvert ! appsink').format(dev, resolution[0], resolution[1])

        # native board camera
        # gst_str = ('nvcamerasrc ! '
        #                 'video/x-raw(memory:NVMM), '
        #                 'width=(int)1280, height=(int)720, '
        #                 'format=(string)I420, framerate=(fraction){}/1 ! '
        #                 'nvvidconv ! '
        #                 'video/x-raw, width=(int){}, height=(int){}, '
        #                 'format=(string)BGRx ! '
        #                 'videoconvert ! appsink').format(framerate,resolution[0], resolution[1])
        self.videoCapture = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        # self.videoCapture = cv2.VideoCapture(1)
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.on = True
        self.frame = None
        print('JetsonCamera loaded... warming camera')
        time.sleep(2)


    def output_from_frame(self, frame):
        # get b,g,r as cv2 uses BGR and not RGB for colors
        # b,g,r = cv2.split(frame)
        # rgb_img = cv2.merge([r,g,b])
        raw = cv2.imencode(".jpg", frame)[1].tostring()

        camera = cv2.imdecode(
            np.fromstring(raw, np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )

        return raw, camera

    def run_threaded(self):
        return self.raw, self.camera

    def run(self):
        _, frame = self.videoCapture.read()
        return self.output_from_frame(frame)

    def update(self):
        print("JetsonCamera update")
        # keep looping infinitely until the thread is stopped
        while self.on:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            _, frame = self.videoCapture.read()
            self.raw, self.camera = self.output_from_frame(frame)
            # if the thread indicator variable is set, stop the thread
            if not self.on:
                break

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        print('stoping JetsonCamera')
        time.sleep(.5)
        self.videoCapture.release ()



WINDOW_NAME = 'CameraDemo'

def read_cam(cap):
    show_help = True
    full_scrn = False
    help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    font = cv2.FONT_HERSHEY_PLAIN
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the window
            # If yes, terminate the program
            break
        _, img = cap.read() # grab the next image frame from camera
        if show_help:
            cv2.putText(img, help_text, (11, 20), font,
                        1.0, (32, 32, 32), 4, cv2.LINE_AA)
            cv2.putText(img, help_text, (10, 20), font,
                        1.0, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(10)
        if key == 27: # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'): # toggle help message
            show_help = not show_help
        elif key == ord('F') or key == ord('f'): # toggle fullscreen
            full_scrn = not full_scrn
            if full_scrn:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)


if __name__ == '__main__':
    print('Called with args:')
    parser = argparse.ArgumentParser(description="")
    print('OpenCV version: {}'.format(cv2.__version__))

    parser.add_argument('--width', dest='image_width',
                        help='image width [160]',
                        default=160, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [120]',
                        default=120, type=int)
    args = parser.parse_args()

    camera = JetsonCamera(resolution=(args.image_width,
                               args.image_height))

    if not camera.videoCapture.isOpened():
        sys.exit('Failed to open camera!')

    open_window(args.image_width, args.image_height)
    img = read_cam(camera.videoCapture)

    camera.videoCapture.release()
    cv2.destroyAllWindows()
