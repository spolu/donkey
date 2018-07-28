import math
import random
import base64
import cv2

import numpy as np
import torch

from simulation import Telemetry
from synthetic import Synthetic, State

EPS = 10e-9

class Engine:
    def __init__(self, config):
        self.frame_stack_size = config.get('frame_stack_size')

        self.step_interval = config.get('synthetic_step_interval')
        self.brake_torque_max = config.get('synthetic_braque_torque_max')
        self.motor_torque_max = config.get('synthetic_motor_torque_max')
        self.speed_max = config.get('synthetic_speed_max')
        self.steer_max = config.get('synthetic_steer_max')

        self.vehicule_mass = config.get('synthetic_vehicule_mass')
        self.vehicule_length = config.get('synthetic_vehicule_length')

        self.wheel_radius = config.get('synthetic_wheel_radius')
        self.wheel_damping_rate = config.get('synthetic_wheel_damping_rate')

        self.track = None
        self.time = None

        self.front_wheel_speed = None  # v_f
        self.steering_angle = None    # delta
        self.heading = None            # theta
        self.front_position = None     # p_f

        self.state_stack = []

        self.synthetic = Synthetic(config)

    def reset(self, track):
        self.track = track
        self.time = 0.0

        self.front_wheel_speed = EPS
        self.steering_angle = EPS
        self.heading = math.pi / 2
        self.front_position = np.array([0.0, 0.0]) + np.array([
            self.vehicule_length * np.cos(self.heading),
            self.vehicule_length * np.sin(self.heading),
        ])

        self.state_stack = [
            self.state() for _ in range(self.frame_stack_size)
        ]

    def start(self, track):
        self.reset(track)

    def stop(self):
        pass

    def step(self, command):
        steering = np.clip(command.steering, -1.0, 1.0)
        throttle = np.clip(command.throttle, 0.0, 1.0)
        brake = np.clip(command.brake, 0.0, 1.0)

        self.time += self.step_interval

        # Integrate v_f.
        wheel_force = (
            throttle * self.motor_torque_max - brake * self.brake_torque_max
        ) / self.wheel_radius

        if self.front_wheel_speed < 0 or self.front_wheel_speed > self.speed_max:
            wheel_force = 0

        self.front_wheel_speed *= 1 - self.wheel_damping_rate
        self.front_wheel_speed += self.step_interval * (
            wheel_force / self.vehicule_mass
        )

        # Immediate steering.
        self.steering_angle = -steering * self.steer_max

        # Integrate heading.
        self.heading += self.step_interval * (
            self.front_wheel_speed /
            self.vehicule_length *
            np.sin(self.steering_angle)
        )

        # Integrate front_position.
        self.front_position += self.step_interval * np.array([
            self.front_wheel_speed * np.cos(self.heading + self.steering_angle),
            self.front_wheel_speed * np.sin(self.heading + self.steering_angle),
        ])

        for i in reversed(range(self.frame_stack_size)):
            if i > 0:
                self.state_stack[i].copy_(self.camera_stack[i-1])
            else:
                self.state_stack[i] = self.state()

    def state(self):
        rear_wheel_speed = self.front_wheel_speed * np.cos(self.steering_angle)

        position = np.array([
            self.front_position[0] - self.vehicule_length * np.cos(self.heading),
            0.0,
            self.front_position[1] - self.vehicule_length * np.sin(self.heading),
        ])
        velocity = np.array([
            rear_wheel_speed * np.cos(self.heading),
            0.0,
            rear_wheel_speed * np.sin(self.heading),
        ])
        angular_velocity = np.array([
            0.0,
            (
                self.front_wheel_speed /
                self.vehicule_length *
                np.sin(self.steering_angle)
            ),
            0.0
        ])

        track_coordinates = self.track.coordinates(position)
        track_angle = self.track.angle(position, velocity, 0)

        return State(
            self.track.randomization,
            position,
            velocity,
            angular_velocity,
            track_coordinates,
            track_angle,
        )

    def telemetry(self):
        camera = self.synthetic.generate(self.state_stack)[0][0]

        return Telemetry(
            self.time,
            camera,
            self.state_stack[0].position,
            self.state_stack[0].velocity,
            self.state_stack[0].angular_velocity,
        )
