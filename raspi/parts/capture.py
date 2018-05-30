import time
import base64
import cv2
import numpy as np
from PIL import Image

from capture import Capture

class Capturer:
    def __init__(self, data_dir, inputs = None, types = None):
        self.capture = Capture(data_dir,load=False)
        self.start_time = time.time()

    def run(self, img_stack = None, accel = None, gyro = None, angle = None, throttle = None, position = None, sense = None):
        '''
        API function needed to use as a Donkey part.
        Accepts values, pairs them with their inputs keys and saves them
        to disk.
        '''
        t = time.time() - self.start_time

        b,g,r = cv2.split(img_stack)       # get b,g,r as cv2 uses BGR and not RGB for colors
        rgb_img = cv2.merge([r,g,b])
        camera = cv2.imencode(".jpg", rgb_img)[1].tostring()

        '''
        vertical vector is [1], keep the vector orientation direct
        '''
        acceleration = np.array([
        accel['y'],
        accel['z'],
        accel['x'],
        ])
        '''
        vertical vector is [1], keep the vector orientation direct
        '''
        angular_velocity = np.array([
        gyro['y'],
        gyro['z'],
        gyro['x'],
        ])

        phone_position = np.array([
        position['x'],
        position['y'],
        position['z'],
        ])

        orientation = np.array([
        sense['roll'],
        sense['yaw'],
        sense['pitch'],
        ])

        self.capture.add_item(
            camera,
            {
            'time': t,
            'raspi_imu_angular_velocity': angular_velocity.tolist(),
            'raspi_imu_acceleration': acceleration.tolist(),
            'raspi_throttle': throttle,
            'raspi_steering': angle,
            'raspi_phone_position': phone_position.tolist(),
            'raspi_sensehat_orientation': orientation.tolist(),
            },
        )

    def shutdown(self):
        pass