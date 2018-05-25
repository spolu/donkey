#!/usr/bin/env python3
"""
Usage:
    donkey.py (drive) [--load_dir=<load_dir>]

Options:
    -h --help        Show this screen.
"""
import os
import argparse

import raspi.vehicle
from utils import Config

#import parts
from raspi.parts.camera import PiCamera
from raspi.parts.cv import ImgStack
from raspi.parts.transform import Lambda
from raspi.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from raspi.parts.runner import Runner
from raspi.parts.web_controller.web import LocalWebController
from raspi.parts.imu import Mpu6050
from raspi.parts.capture import Capturer

# VEHICLE
DRIVE_LOOP_HZ = 10
MAX_LOOPS = 100000

# CAMERA
CAMERA_RESOLUTION = (120, 160) #(height, width)
CAMERA_FRAMERATE = DRIVE_LOOP_HZ

# STEERING
STEERING_CHANNEL = 0
#STEERING_LEFT_PWM = 440
STEERING_LEFT_PWM = 425
STEERING_RIGHT_PWM = 342

# THROTTLE
THROTTLE_CHANNEL = 1
#THROTTLE_FORWARD_PWM = 420
THROTTLE_FORWARD_PWM = 400
THROTTLE_STOPPED_PWM = 360
THROTTLE_REVERSE_PWM = 320
#THROTTLE_REVERSE_PWM = 310

def drive(args):
    
    #Initialize car
    V = raspi.vehicle.Vehicle()
    cam = PiCamera(resolution=CAMERA_RESOLUTION)
    V.add(cam, outputs=['cam/image_array'], threaded=True)

    imu = Mpu6050()
    V.add(imu, outputs=['imu/acl', 'imu/gyr'], threaded=True)

    stack = ImgStack()
    V.add(stack,
          inputs=['cam/image_array'],
          outputs=['cam/image_stack'],
          threaded=False)

    ctr = LocalWebController()
    V.add(ctr,
          inputs=['cam/image_array'],
          outputs=['angle', 'throttle'],
          threaded=True)

    if args.capture_dir is not None:
        capturer = Capturer(args.capture_dir)
        V.add(capturer,
              inputs=['cam/image_array','imu/acl', 'imu/gyr', 'angle', 'throttle'],
              threaded=False)

    steering_controller = PCA9685(STEERING_CHANNEL)
    steering = PWMSteering(controller=steering_controller,
                                    left_pulse=STEERING_LEFT_PWM,
                                    right_pulse=STEERING_RIGHT_PWM)

    throttle_controller = PCA9685(THROTTLE_CHANNEL)
    throttle = PWMThrottle(controller=throttle_controller,
                                    max_pulse=THROTTLE_FORWARD_PWM,
                                    zero_pulse=THROTTLE_STOPPED_PWM,
                                    min_pulse=THROTTLE_REVERSE_PWM)

    V.add(steering, inputs=['angle'])
    V.add(throttle, inputs=['throttle'])

    V.start(rate_hz=DRIVE_LOOP_HZ,
            max_loop_count=MAX_LOOPS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--capture_dir', type=str, help="path to save training data")

    args = parser.parse_args()

    drive(args)
