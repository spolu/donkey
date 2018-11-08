import numpy as np
import time

from reinforce.input_filter import InputFilter


class Dummy:
    def __init__(self, cfg):
        self.driver_fixed_throttle = cfg.get('driver_fixed_throttle')
        self.driver_optical_flow_speed = cfg.get('driver_optical_flow_speed')
        self.config = cfg

        self.input_filter = InputFilter(cfg)

        self.start_time = time.time()
        self.throttle = self.driver_fixed_throttle

    def run(self, camera=None, flow_speed=None):
        camera = self.input_filter.apply(camera) / 127.5

        left = np.sum(camera[15:, :80])
        right = np.sum(camera[15:, 80:])
        p = left / (left + right)
        steering = (p-0.5) * 4

        throttle = self.throttle

        SPEED = self.driver_optical_flow_speed

        if time.time() - self.start_time > 10.0:
            if flow_speed < SPEED-5:
                self.throttle += 0.001
            elif flow_speed < SPEED:
                self.throttle += 0.0005

            if flow_speed > SPEED:
                self.throttle -= 0.0005
            elif flow_speed > SPEED+5:
                self.throttle -= 0.001

        print(">>> COMMANDS: {:.2f} {:.2f}".format(
            steering, throttle
        ))

        return steering, throttle
