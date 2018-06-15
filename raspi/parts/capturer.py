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

    def update(self):
        pass

    def run(
            self,
            angle = None,
            throttle = None,
            cam_img_array = None,
            imu_accel = None,
            imu_gyro = None,
            imu_stack = None,
            sense_orientation = None,
            pozyx_position = None,
            pozyx_stack = None,
    ):
        '''
        API function needed to use as a Donkey part.
        Accepts values, pairs them with their inputs keys and saves them
        to disk.
        '''
        t = time.time() - self.start_time

        b,g,r = cv2.split(cam_img_array)       # get b,g,r as cv2 uses BGR and not RGB for colors
        rgb_img = cv2.merge([r,g,b])
        camera = cv2.imencode(".jpg", rgb_img)[1].tostring()

        '''
        vertical vector is [1], keep the vector orientation direct
        '''
        acceleration = np.array([
            imu_accel['y'],
            imu_accel['z'],
            imu_accel['x'],
        ])
        '''
        vertical vector is [1], keep the vector orientation direct
        '''
        angular_velocity = np.array([
            imu_gyro['y'],
            imu_gyro['z'],
            imu_gyro['x'],
        ])

        position = np.array([
            pozyx_position['x'],
            pozyx_position['y'],
            pozyx_position['z'],
        ])

        orientation = np.array([
            sense_orientation['roll'],
            sense_orientation['yaw'],
            sense_orientation['pitch'],
        ])

        items = []

        for r in imu_stack:
            acceleration = np.array([
                r['accel']['y'],
                r['accel']['z'],
                r['accel']['x'],
            ])
            angular_velocity = np.array([
                r['gyro']['y'],
                r['gyro']['z'],
                r['gyro']['x'],
            ])
            items.append({
                'time': r['time'] - self.start_time,
                'raspi_imu_angular_velocity': angular_velocity.tolist(),
                'raspi_imu_acceleration': acceleration.tolist(),
            })

        for r in pozyx_stack:
            position = np.array([
                r['position']['x'],
                r['position']['y'],
                r['position']['z'],
            ])
            items.append({
                'time': r['time'] - self.start_time,
                'raspi_pozyx_position': position.tolist(),
            })

        items.sort(key=lambda it: it['time'])

        for it in items:
            self.capture.add_item(
                None, it, save=False
            )

        self.capture.add_item(
            camera,
            {
                'time': t,
                'raspi_throttle': throttle,
                'raspi_steering': angle,
                'raspi_imu_angular_velocity': angular_velocity.tolist(),
                'raspi_imu_acceleration': acceleration.tolist(),
                'raspi_pozyx_position': position.tolist(),
                'raspi_sensehat_orientation': orientation.tolist(),
            },
            save=False,
        )

    def shutdown(self):
        self.capture.save()
