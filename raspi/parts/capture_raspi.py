import time
import base64
import cv2
import numpy as np
from PIL import Image

from capture import Capture

class CaptureRaspi:
    def __init__(self, data_dir, inputs = None, types = None):
        self.capture = Capture(data_dir,load=False)
        self.start_time = time.time()

    def run(self, img_stack = None, accel_x = None, accel_y = None, gyro_z = None):
        '''
        API function needed to use as a Donkey part.
        Accepts values, pairs them with their inputs keys and saves them
        to disk.
        '''
        t = time.time()
        camera = cv2.imencode(".jpg", img_stack)[1].tostring()
        planar_acceleration = np.linalg.norm([accel_x,accel_y])

        self.capture.add_item(
            camera,
            {
            'time': t,
            'angular_velocity': gyro_z,
            'acceleration': planar_acceleration,
            },
        )

    def shutdown(self):
        pass