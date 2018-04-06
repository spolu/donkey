#!/usr/bin/env python3
"""
Usage:
    donkey.py (drive) [--model=<model>]

Options:
    -h --help        Show this screen.
"""
import os
from docopt import docopt

import vehicle
import config

#import parts
from parts.camera import PiCamera
from parts.transform import Lambda
from parts.actuator import PCA9685, PWMSteering, PWMThrottle
# from parts.human import Human
from parts.web_controller.web import LocalWebController

def drive(cfg, model_path=None):
    #Initialize car
    V = vehicle.Vehicle()
    cam = PiCamera(resolution=cfg.CAMERA_RESOLUTION)
    V.add(cam, outputs=['cam/image_array'], threaded=True)

    if model_path is None:
        ctr = LocalWebController()

    V.add(ctr,
          inputs=['cam/image_array'],
          outputs=['angle', 'throttle'],
          threaded=True)

    steering_controller = PCA9685(cfg.STEERING_CHANNEL)
    steering = PWMSteering(controller=steering_controller,
                                    left_pulse=cfg.STEERING_LEFT_PWM,
                                    right_pulse=cfg.STEERING_RIGHT_PWM)

    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL)
    throttle = PWMThrottle(controller=throttle_controller,
                                    max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                    zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                                    min_pulse=cfg.THROTTLE_REVERSE_PWM)

    V.add(steering, inputs=['angle'])
    V.add(throttle, inputs=['throttle'])

    print("You can now go to d2.local:8887 to drive.")

    #run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)

if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = config.load_config("config_defaults.py")

    if args['drive']:
        drive(cfg, model_path = args['--model'])
