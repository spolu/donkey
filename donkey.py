import simulation
import track
import base64
import cv2
import numpy as np


MAX_GAME_TIME = 120
OFF_TRACK_DISTANCE = 6.0
CAMERA_SIZE = 120 * 160

class Donkey:
    def __init__(self, headless=True):
        self.started = False
        self.simulation = simulation.Simulation(headless=headless)
        self.track = track.Track()

    def observation_from_telemetry(self, telemetry):
        """
        Returns a float numpy array of size 1x(CAMERA_SIZE+3+3):
        - the camera black and white image
        - the current car velocity
        - the current car acceleration
        """
        camera = cv2.imdecode(
            np.fromstring(base64.b64decode(telemetry['camera']), np.uint8),
            cv2.IMREAD_GRAYSCALE,
        ).astype(np.float)
        camera = camera / 255.0
        camera = camera.reshape(1,CAMERA_SIZE)

        velocity = np.array([[
            telemetry['velocity']['x'],
            telemetry['velocity']['y'],
            telemetry['velocity']['z'],
        ]])

        acceleration = np.array([[
            telemetry['acceleration']['x'],
            telemetry['acceleration']['y'],
            telemetry['acceleration']['z'],
        ]])

        return np.concatenate((camera, velocity, acceleration), axis=1)

    def reward_from_telemetry(self, telemetry):
        position = np.array([
            telemetry['position']['x'],
            telemetry['position']['y'],
            telemetry['position']['z'],
        ])
        velocity = np.array([
            telemetry['velocity']['x'],
            telemetry['velocity']['y'],
            telemetry['velocity']['z'],
        ])
        return self.track.speed(position, speed)

    def done_from_telemetry(self, telemetry):
        if telemetry['time'] > MAX_GAME_TIME:
            return True
        position = np.array([
            telemetry['position']['x'],
            telemetry['position']['y'],
            telemetry['position']['z'],
        ])
        if self.track.distance(position) > OFF_TRACK_DISTANCE:
            return True
        return False

    def reset(self):
        """
        `reset` resets the environment to its initial state and returns an
        initial observation (see `step`). `reset` must be called at least once
        before calling `step`.
        """
        if not self.started:
            self.started = True
            self.simulation.start()
        else:
            self.simulation.reset()
        telemetry = self.simulation.telemetry()

        observation = self.observation_from_telemetry(telemetry)

        return observation

    def step(self, controls):
        """
        `step` takes as input a 1x3 float numpy array. Its values are clamped
        to their valid intervals:
        - steering is clamped to [-1;1]
        - throttle is clamped to [0;1]
        - brake is clamped to [0;1]
        It returns:
        - an observation float numpy array of size 1x(CAMERA_SIZE+3+3):
          - the camera black and white image
          - the current car velocity
          - the current car acceleration
        - a reward value for the last step
        - a boolean indicating whether the game is finished
        """
        steering = np.clip(controls[0][0], -1, 1)
        throttle = np.clip(controls[0][0], 0, 1)
        brake = np.clip(controls[0][0], 0, 1)

        command = simulation.Command(steering, throttle, brake)

        c.step(command)
        telemetry = self.simulation.telemetry()

        observation = self.observation_from_telemetry(telemetry)
        reward = self.reward_from_telemetry(telemetry)
        done = self.done_from_telemetry(telemetry)

        return observation, reward, done
