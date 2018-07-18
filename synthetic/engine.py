import math
import random

import numpy as np

class Engine:
    def __init__(self, config):
        self.step_interval = config.get('engine_step_interval')
        self.brake_torque_max = config.get('engine_braque_torque_max')
        self.motor_torque_max = config.get('engine_motor_torque_max')
        self.speed_max = config.get('engine_speed_max')
        self.steer_max = config.get('engine_steer_max')
        self.wheel_damping_rate = config.get('engine_wheel_damping_rate')
        self.vehicule_mass = config.get('engine_vehicule_mass')
        self.vehicule_length = config.get('engine_vehicule_length')

        self.track = None
        self.time = None

        self.steeering_angle = None    # delta
        self.heading = None            # theta

        self.front_position = None     # p_f
        self.front_velocity = None     # dp_f/dt

        self.front_wheel_speed = None  # v_f

    def reset(self, track):
        self.track = track
        self.time = 0.0

        self.steeering_angle = 0.0
        self.heading = 0.0

        self.front_position = np.array([0.0, 0.0]) + np.array([
            self.vehicule_length * np.cos(self.heading),
            self.vehicule_length * np.sin(self.heading),
        ])
        self.front_velocity = np.array([0.0, 0.0])

        self.front_wheel_speed = 0.0

    def step(self, command):
        steering = np.clip(command.steering, -1.0, 1.0)
        throttle = np.clip(command.throttle, 0.0, 1.0)
        brake = np.clip(command.brake, 0.0, 1.0)

        pass
