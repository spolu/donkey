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
        self.window = []

    def run(self, camera=None, flow_speed=None):
        camera = self.input_filter.apply(camera) / 127.5

        left = np.sum(camera[15:, :80])
        right = np.sum(camera[15:, 80:])
        p = left / (left + right)

        self.window.append([p, time.time()])
        self.window = self.window[-4:]

        if len(self.window) == 4:
            dps = [
                (self.window[i+1][0]-self.window[i][0]) /
                (self.window[i+1][1]-self.window[i][1])
                for i in range(3)
            ]
            dp = sum(dps) / len(dps)
            pre = ((p + dp/10) - 0.5)
            if abs(pre) > 0.25:
                steering = pre * 4
            else:
                steering = pre * 2
            print("DEBUG: p={} dp={} steering={}".format(p, dp, steering))
        else:
            steering = (p-0.5) * 4
            print("DEBUG: p={}".format(p))

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
