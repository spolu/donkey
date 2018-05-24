import time
import base64
import cv2
import numpy as np
from PIL import Image

from capture import Capture

class CaptureRaspi:
    def __init__(self, data_dir, inputs = None, types = None):
        self.capture = Capture(data_dir)
        self.start_time = time.time()

    def run(self, img_stack = None, gyro_z = None):
        '''
        API function needed to use as a Donkey part.
        Accepts values, pairs them with their inputs keys and saves them
        to disk.
        '''
        print('run capture raspi step')
        print(img_stack)
        print(gyro_z)
        t = time.time()
        camera = im = Image.fromarray(img_stack)
        
        self.capture.add_item(
            camera,
            {
            'time': t,
            'angular_velocity': gyro_z,
            },
        )


    def shutdown(self):
        pass