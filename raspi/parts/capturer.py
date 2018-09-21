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
            camera_raw = None,
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

        item = {
            'time': t,
            'raspi_throttle': throttle,
            'raspi_steering': angle,
        }

        if imu_accel is not None:
            # vertical vector is [1], keep the vector orientation direct
            acceleration = np.array([
                imu_accel['y'],
                imu_accel['z'],
                imu_accel['x'],
            ])
            item['raspi_imu_acceleration'] = acceleration.tolist()

        if imu_gyro is not None:
            # vertical vector is [1], keep the vector orientation direct
            angular_velocity = np.array([
                imu_gyro['y'],
                imu_gyro['z'],
                imu_gyro['x'],
            ])
            item['raspi_imu_angular_velocity'] = angular_velocity.tolist()

        if pozyx_position is not None:
            position = np.array([
                pozyx_position['x'],
                pozyx_position['y'],
                pozyx_position['z'],
            ])
            item['raspi_pozyz_position'] = position.tolist()

        if sense_orientation is not None:
            orientation = np.array([
                sense_orientation['roll'],
                sense_orientation['yaw'],
                sense_orientation['pitch'],
            ])
            item['raspi_sensehat_orientation'] = orientation.tolist()

        items = []

        if imu_stack is not None:
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
        if pozyx_stack is not None:
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
            self.capture.add_item(None, it, save=False)
        self.capture.add_item(camera_raw, item, save=False)

    def shutdown(self):
        self.capture.save()
