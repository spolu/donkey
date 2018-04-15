import simulation
import track
import base64
import collections
import cv2
import numpy as np
import random
import math

from eventlet.green import threading

MAX_SPEED = 10.0

STALL_SPEED = 0.1
MAX_STALL_TIME = 10

OFF_TRACK_DISTANCE = 6.0

LAP_COUNT = 3
PROGRESS_INCREMENT = 0.01
REFERENCE_LAP_TIME = 100.0

CAMERA_STACK_SIZE = 3
CAMERA_WIDTH = 120
CAMERA_HEIGHT = 160

CONTINUOUS_CONTROL_SIZE = 2
DISCRETE_CONTROL_SIZE = 9

ANGLES_WINDOW = 4

Observation = collections.namedtuple(
    'Observation',
    'progress, time, track_angles, track_position, track_linear_speed, position, velocity, acceleration, camera'
)

class Donkey:
    def __init__(self, config):
        self.track_name = config.get('track_name')
        self.track_randomized = config.get('track_randomized')
        self.reward_type = config.get('reward_type')
        self.action_type = config.get('action_type')
        self.speed_limit = config.get('speed_limit')

        self.simulation_headless = config.get('simulation_headless')
        self.simulation_time_scale = config.get('simulation_time_scale')
        self.simulation_step_interval = config.get('simulation_step_interval')
        self.simulation_capture_frame_rate = config.get('simulation_capture_frame_rate')

        self.started = False

        self.track = track.Track(self.track_name)
        if self.track_randomized:
            self.track.randomize()

        self.simulation = simulation.Simulation(
            True,
            self.simulation_headless,
            self.simulation_time_scale,
            self.simulation_step_interval,
            self.simulation_capture_frame_rate,
        )

        self.last_reset_time = 0.0
        self.step_count = 0
        self.lap_count = 0
        self.last_lap_time = 0.0
        self.last_progress = 0.0
        self.last_unstall_time = 0.0
        self.last_rewarded_progress = 0.0
        self.last_track_linear_speed = 0.0

        self.camera_stack = None

    def observation_from_telemetry(self, telemetry):
        """
        Returns a named tuple with physical measurements as well as camera.
        """
        camera = cv2.imdecode(
            np.fromstring(base64.b64decode(telemetry['camera']), np.uint8),
            cv2.IMREAD_GRAYSCALE,
        ).astype(np.float)

        # Scale, size is 120x160.
        camera = camera / 127.5 - 1

        if self.camera_stack is None:
            self.camera_stack = np.zeros((CAMERA_STACK_SIZE, CAMERA_WIDTH, CAMERA_HEIGHT))
            for i in range(CAMERA_STACK_SIZE):
                self.camera_stack[i] = np.copy(camera)
        else:
            for i in reversed(range(CAMERA_STACK_SIZE)):
                if i > 0:
                    self.camera_stack[i] = self.camera_stack[i-1]
                else:
                    self.camera_stack[i] = np.copy(camera)

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
        acceleration = np.array([
            telemetry['acceleration']['x'],
            telemetry['acceleration']['y'],
            telemetry['acceleration']['z'],
        ])

        track_angles = []
        for i in range(ANGLES_WINDOW):
            track_angles.append(self.track.angle(position, velocity, i) / math.pi)

        track_position = self.track.position(position) / OFF_TRACK_DISTANCE
        track_linear_speed = self.track.linear_speed(position, velocity) / MAX_SPEED
        self.last_track_linear_speed = track_linear_speed

        progress = self.track.progress(position) / self.track.length
        time = telemetry['time'] - self.last_reset_time

        return Observation(
            progress,
            time,
            track_angles,
            track_position,
            track_linear_speed,
            position,
            velocity,
            acceleration,
            np.copy(self.camera_stack),
        )

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

        progress = self.track.progress(position) / self.track.length
        time = telemetry['time'] - self.last_reset_time

        track_linear_speed = self.track.linear_speed(position, velocity)
        track_lateral_speed = self.track.lateral_speed(position, velocity)
        track_position = self.track.position(position)

        if self.reward_type == "speed":
            return (
                2 * track_linear_speed -
                track_lateral_speed -
                np.linalg.norm(track_position)
            ) / (MAX_SPEED * OFF_TRACK_DISTANCE)

        if self.reward_type == "speed_cap":
            if track_linear_speed > 2.0:
                return (
                    - (track_linear_speed - 1.0) -
                    track_lateral_speed -
                    np.linalg.norm(track_position)
                ) / (MAX_SPEED * OFF_TRACK_DISTANCE)
            else:
                return (
                    4 * track_linear_speed -
                    track_lateral_speed -
                    np.linalg.norm(track_position)
                ) / (MAX_SPEED * OFF_TRACK_DISTANCE)


        if self.reward_type == "time":
            if (progress - self.last_rewarded_progress) > PROGRESS_INCREMENT:
                self.last_rewarded_progress = progress
                return 1.0 - (time - self.last_lap_time) / (progress * REFERENCE_LAP_TIME)
            return 0.0

        return 0.0

    def done_from_telemetry(self, telemetry):
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

        # If we're off track, stop.
        track_position = self.track.position(position)
        if np.linalg.norm(track_position) > OFF_TRACK_DISTANCE:
            return True

        # If we stall (STALL_SPEED) for more than MAX_STALL_TIME then stop.
        track_linear_speed = self.track.linear_speed(position, velocity)
        time = telemetry['time'] - self.last_reset_time
        if track_linear_speed > STALL_SPEED:
            self.last_unstall_time = time
        elif (time - self.last_unstall_time > MAX_STALL_TIME):
            return True

        # If the last progress is bigger than the current one, it means we just
        # crossed the finish line, stop.
        progress = self.track.progress(position) / self.track.length
        if self.last_progress > progress + 0.5:
            print("LAP TIME: {}".format(time - self.last_lap_time))
            if self.lap_count == 2:
                return True
            self.lap_count += 1
            self.last_lap_time = time
            self.last_rewarded_progress = 0.0
        self.last_progress = progress

        return False

    def reset(self):
        """
        `reset` resets the environment to its initial state and returns an
        initial observation (see `observation_from_telemetry`). `reset` must be
        called at least once before calling `step`.
        """
        if self.track_randomized:
            self.track.randomize()

        if not self.started:
            self.started = True
            self.simulation.start(self.track)
        else:
            if random.randint(1, 50) == 1:
                self.simulation.stop()
                self.simulation.start(self.track)
            else:
                self.simulation.reset(self.track)
        telemetry = self.simulation.telemetry()
        self.last_reset_time = telemetry['time']

        self.lap_count = 0
        self.last_lap_time = 0.0
        self.last_progress = 0.0
        self.last_unstall_time = 0.0
        self.last_rewarded_progress = 0.0
        self.last_track_linear_speed = 0.0

        self.camera_stack = None

        observation = self.observation_from_telemetry(telemetry)
        # print("TELEMETRY RESET {}".format(telemetry))

        return observation

    def step(self, controls):
        """
        Runs a step of the simuation based on input.
        It returns:
        - the observation as computed by `observation_from_telemetry`
        - a reward value for the last step
        - a boolean indicating whether the game is finished
        """
        if self.action_type == 'discrete':
            if controls == 0:
                throttle = 0.0
                brake = 0.0
                steering = 0.0
            if controls == 1:
                throttle = 0.5
                brake = 0.0
                steering = 0.0
            if controls == 2:
                throttle = 1.0
                brake = 0.0
                steering = 0.0
            if controls == 3:
                throttle = 0.0
                brake = 0.5
                steering = 0.0
            if controls == 4:
                throttle = 0.0
                brake = 1.0
                steering = 0.0
            if controls == 5:
                throttle = 0.0
                brake = 0.0
                steering = 0.5
            if controls == 6:
                throttle = 0.0
                brake = 0.0
                steering = 1.0
            if controls == 7:
                throttle = 0.0
                brake = 0.0
                steering = -0.5
            if controls == 8:
                throttle = 0.0
                brake = 0.0
                steering = -1.0
        else:
            steering = controls[0]
            throttle_brake = controls[1]

            throttle= 0.0
            brake = 0.0

            if throttle_brake > 0.0:
                throttle = throttle_brake
                brake = 0.0
            if throttle_brake < 0.0:
                throttle = 0.0
                brake = -throttle_brake

        if self.speed_limit > 0.0:
            if self.last_track_linear_speed > self.speed_limit:
                throttle = 0.0

        command = simulation.Command(steering, throttle, brake)

        self.simulation.step(command)
        telemetry = self.simulation.telemetry()

        observation = self.observation_from_telemetry(telemetry)
        reward = self.reward_from_telemetry(telemetry)
        done = self.done_from_telemetry(telemetry)

        self.step_count += 1

        if done:
            self.reset()
            # If we're done we read the new observations post reset.
            observation = self.observation_from_telemetry(telemetry)

        return observation, reward, done


_send_condition = threading.Condition()
_recv_condition = threading.Condition()
_recv_count = 0

class Worker(threading.Thread):
    def __init__(self, config):
        self.condition = threading.Condition()
        self.controls = None
        self.observation = None
        self.reward = 0.0
        self.done = False
        self.donkey = Donkey(config)
        threading.Thread.__init__(self)

    def reset(self):
        self.controls = None
        self.reward = 0.0
        self.done = False
        self.observation = self.donkey.reset()

    def run(self):
        global _recv_count
        global _send_condition
        global _recv_condition
        while True:
            # Wait for the controls to be set.
            _send_condition.acquire()
            _send_condition.wait()
            _send_condition.release()

            observation, reward, done = self.donkey.step(
                self.controls
            )

            self.observation = observation
            self.reward = reward
            self.done = done

            # Notify that we are done.
            _recv_condition.acquire()
            _recv_count = _recv_count + 1
            _recv_condition.notify_all()
            _recv_condition.release()

class Envs:
    def __init__(self, config):
        self.worker_count = config.get('worker_count')
        self.workers = [Worker(config) for _ in range(self.worker_count)]
        for w in self.workers:
            w.start()

    def reset(self):
        for w in self.workers:
            w.reset()
        observations = [w.observation for w in self.workers]

        return observations

    def step(self, controls):
        global _recv_count
        global _send_condition
        global _recv_condition

        _recv_condition.acquire()
        _recv_count = 0

        for i in range(len(self.workers)):
            w = self.workers[i]
            w.controls = controls[i]

            # Release the workers.
            _send_condition.acquire()
            _send_condition.notify()
            _send_condition.release()

        # Wait for the workers to finish.
        first = True
        while _recv_count < len(self.workers):
            if first:
                first = False
            else:
                _recv_condition.acquire()
            _recv_condition.wait()
            _recv_condition.release()

        dones = [w.done for w in self.workers]
        rewards = [w.reward for w in self.workers]
        observations = [w.observation for w in self.workers]

        return observations, rewards, dones
