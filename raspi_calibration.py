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
from track import Track

#import parts
from raspi.parts.camera import PiCamera
from raspi.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from raspi.parts.camera_flow import CameraFlow
from raspi.parts.driver import Driver
from raspi.parts.dummy import Dummy
from raspi.parts.sense import Sense
from raspi.parts.capturer import Capturer
from raspi.parts.web_controller.web import LocalWebController

# VEHICLE
DRIVE_LOOP_HZ = 30
MAX_LOOPS = 100000

# CAMERA
CAMERA_RESOLUTION = (480, 640) #(height, width)
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
    cfg = Config(args.config_path)

    cfg.override('cuda', False)
    if args.reinforce_load_dir != None:
        cfg.override('reinforce_load_dir', args.reinforce_load_dir)
    if args.driver_fixed_throttle != None:
        cfg.override('driver_fixed_throttle', args.driver_fixed_throttle)
    if args.driver_optical_flow_speed != None:
        cfg.override('driver_optical_flow_speed', args.driver_optical_flow_speed)
    if args.canny_low != None:
        cfg.override('input_filter_canny_low', args.canny_low)
    if args.canny_high != None:
        cfg.override('input_filter_canny_high', args.canny_high)

    #Initialize car
    V = raspi.vehicle.Vehicle()

    cam = PiCamera(resolution=CAMERA_RESOLUTION)
    V.add(
        cam,
        outputs=[
            'cam/raw',
            'cam/camera',
        ],
        threaded=True,
    )

    if args.capture_dir is not None:
        capturer = Capturer(args.capture_dir)
        V.add(
            capturer,
            inputs=[
                'angle',
                'throttle',
                # 'dp',
                'cam/raw',
            ],
            threaded=False,
        )

    V.start(rate_hz=DRIVE_LOOP_HZ,
            max_loop_count=MAX_LOOPS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('config_path', type=str, help="path to the config file")

    parser.add_argument('--reinforce_load_dir', type=str, help="config override")
    parser.add_argument('--capture_dir', type=str, help="path to save training data")

    parser.add_argument('--driver_fixed_throttle', type=float, help="config override")
    parser.add_argument('--driver_optical_flow_speed', type=float, help="config override")
    parser.add_argument('--canny_low', type=int, help="low in canny parameters, all points below are not detected")
    parser.add_argument('--canny_high', type=int, help="high in canny parameters, all points above are detected")

    args = parser.parse_args()

    drive(args)
