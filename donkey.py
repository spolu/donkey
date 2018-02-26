import simulation
import track
import base64
import cv2
import numpy as np


MAX_GAME_TIME = 120
OFF_TRACK_DISTANCE = 6.0
CAMERA_CHANNEL = 3
CAMERA_WIDTH = 120
CAMERA_HEIGHT = 160
CONTROL_SIZE = 3
SIMULATION_TIME_SCALE = 40.0
SIMULATION_STEP_INTERVAL = 0.10

class Donkey:
    def __init__(self, headless=True):
        self.started = False
        self.simulation = simulation.Simulation(
            True, headless, SIMULATION_TIME_SCALE, SIMULATION_STEP_INTERVAL,
        )
        self.track = track.Track()
        self.last_reset_time = 0.0
        self.step_count = 0

    def observation_from_telemetry(self, telemetry):
        """
        Returns a tuple of float numpy arrays:
        - the camera color image (3x120x160)
        - the current car velocity (3D)
        - the current car acceleration (3D)
        """
        camera = cv2.imdecode(
            np.fromstring(base64.b64decode(telemetry['camera']), np.uint8),
            cv2.IMREAD_COLOR,
        ).astype(np.float)

        # Scale and transpose to 3x120x160.
        camera = np.transpose(camera / 255.0, (2, 0, 1))

        # velocity = np.array([
        #     telemetry['velocity']['x'],
        #     telemetry['velocity']['y'],
        #     telemetry['velocity']['z'],
        # ])

        # acceleration = np.array([
        #     telemetry['acceleration']['x'],
        #     telemetry['acceleration']['y'],
        #     telemetry['acceleration']['z'],
        # ])

        # return (camera, velocity, acceleration)
        return camera

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

        speed =  self.track.speed(position, velocity)
        # print("SPEED {}".format(speed))

        if speed > 1.0:
            return 1.0
        if speed < 0.0:
            return -1.0

        return 0.0

    def done_from_telemetry(self, telemetry):
        if (telemetry['time'] - self.last_reset_time) > MAX_GAME_TIME:
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
        initial observation (see `observation_from_telemetry`). `reset` must be
        called at least once before calling `step`.
        """
        if not self.started:
            self.started = True
            self.simulation.start()
        else:
            self.simulation.reset()
        telemetry = self.simulation.telemetry()
        self.last_reset_time = telemetry['time']

        observation = self.observation_from_telemetry(telemetry)

        return observation

    def step(self, controls):
        """
        `step` takes as input a 3D float numpy array. Its values are clamped
        to their valid intervals:
        - steering is clamped to [-1;1]
        - throttle is clamped to [0;1]
        - brake is clamped to [0;1]
        It returns:
        - the observation as computed by `observation_from_telemetry`
        - a reward value for the last step
        - a boolean indicating whether the game is finished
        """
        steering = np.clip(controls[0], -1, 1)
        throttle = np.clip(controls[1], 0, 1)
        brake = np.clip(controls[2], 0, 1)

        # print("RECEIVED  {} {} {}".format(steering, throttle, brake))

        command = simulation.Command(steering, throttle, brake)

        self.simulation.step(command)
        telemetry = self.simulation.telemetry()

        observation = self.observation_from_telemetry(telemetry)
        reward = self.reward_from_telemetry(telemetry)
        done = self.done_from_telemetry(telemetry)

        if self.step_count % 1000 == 0:
            print("TELEMETRY {}".format(telemetry))
        self.step_count += 1


        if done:
            self.reset()

        return observation, reward, done
