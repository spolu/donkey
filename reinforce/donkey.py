import base64
import cv2
import random
import math
import os

import numpy as np

from eventlet.green import threading

from simulation import Command, Observation, Telemetry
from track import Track
from unity import Unity
from synthetic import Engine
from capture import Capture


MAX_SPEED = 10.0
STALL_SPEED = 0.1
MAX_STALL_TIME = 10

MAX_OUTTRACK_TIME = 0

LAP_COUNT = 3
PROGRESS_INCREMENT = 0.01
REFERENCE_LAP_TIME = 100.0

CAMERA_STACK_SIZE = 3
CAMERA_HEIGHT = 120
CAMERA_WIDTH = 160

CONTINUOUS_CONTROL_SIZE = 2
DISCRETE_THROTTLE_SLOW = 0.6
DISCRETE_CONTROL_COUNT = 9

ANGLES_WINDOW = 5

class Donkey:
    def __init__(self, config, save_dir=None, worker_index=0):
        self.track_name = config.get('track_name')
        self.track_randomized = config.get('track_randomized')
        self.reward_type = config.get('reward_type')
        self.action_type = config.get('action_type')
        self.speed_limit = config.get('speed_limit')

        self.unity_headless = config.get('unity_headless')
        self.unity_time_scale = config.get('unity_time_scale')
        self.unity_step_interval = config.get('unity_step_interval')
        self.unity_capture_frame_rate = config.get('unity_capture_frame_rate')

        self.worker_index = worker_index
        self.do_capture = save_dir != None and config.get('do_capture')

        if self.do_capture:
            self.capture_set_dir = os.path.join(save_dir, 'capture_set')
            if not os.path.isdir(self.capture_set_dir):
                os.mkdir(self.capture_set_dir)

            self.capture_index = 0
            self.capture = Capture(
                os.path.join(
                    self.capture_set_dir,
                    str(self.worker_index) + '-' + str(self.capture_index),
                ),
                load=False,
            )

        self.started = False

        self.track = Track(self.track_name)
        if self.track_randomized:
            self.track.randomize()

        self.simulation = Unity(
            True,
            self.unity_headless,
            self.unity_time_scale,
            self.unity_step_interval,
            self.unity_capture_frame_rate,
            None,
        )

        self.last_reset_time = 0.0
        self.step_count = 0
        self.lap_count = 0
        self.last_lap_time = 0.0
        self.last_progress = 0.0
        self.last_unstall_time = 0.0
        self.last_intrack_time = 0.0
        self.last_rewarded_advance = 0.0
        self.last_track_linear_speed = 0.0

        self.camera_stack = None

    def observation_from_telemetry(self, telemetry):
        """
        Returns a named tuple with physical measurements as well as camera.
        """
        camera_raw = base64.b64decode(telemetry.camera)
        camera = cv2.imdecode(
            np.fromstring(camera_raw, np.uint8),
            cv2.IMREAD_GRAYSCALE,
        ).astype(np.float)

        # Scale, size is 120x160.
        camera = camera / 127.5 - 1

        if self.camera_stack is None:
            self.camera_stack = np.zeros(
                (CAMERA_STACK_SIZE, CAMERA_HEIGHT, CAMERA_WIDTH),
            )
            for i in range(CAMERA_STACK_SIZE):
                self.camera_stack[i] = np.copy(camera)
        else:
            for i in reversed(range(CAMERA_STACK_SIZE)):
                if i > 0:
                    self.camera_stack[i] = self.camera_stack[i-1]
                else:
                    self.camera_stack[i] = np.copy(camera)

        position = telemetry.position
        velocity = telemetry.velocity
        angular_velocity = telemetry.angular_velocity

        track_angles = []
        for i in range(ANGLES_WINDOW):
            # TODO(stan): remove math.pi regularization
            track_angles.append(self.track.angle(position, velocity, i) / math.pi)

        track_coordinates = self.track.coordinates(position)

        track_linear_speed = self.track.linear_speed(position, velocity) / MAX_SPEED
        self.last_track_linear_speed = track_linear_speed

        time = telemetry.time - self.last_reset_time

        return Observation(
            time,
            track_coordinates,
            track_angles,
            track_linear_speed,
            position,
            velocity,
            angular_velocity,
            np.copy(camera),
            np.copy(self.camera_stack),
            camera_raw,
        )

    def reward_from_telemetry(self, telemetry):
        position = telemetry.position
        velocity = telemetry.velocity

        time = telemetry.time - self.last_reset_time

        track_coordinates = self.track.coordinates(position)
        track_position = self.track.position(track_coordinates)
        track_progress = self.track.progress(track_coordinates)

        track_linear_speed = self.track.linear_speed(position, velocity)
        track_lateral_speed = self.track.lateral_speed(position, velocity)

        if self.reward_type == "speed":
            return (
                2 * track_linear_speed -
                track_lateral_speed -
                np.linalg.norm(track_position) -
                10 * max(0, np.linalg.norm(track_position) - 1.0)
            ) / MAX_SPEED

        if self.reward_type == "speed_cap":
            if track_linear_speed > 2.0:
                return (
                    - (track_linear_speed - 1.0) -
                    track_lateral_speed -
                    np.linalg.norm(track_position) -
                    10 * max(0, np.linalg.norm(track_position) - 1.0)
                ) / MAX_SPEED
            else:
                return (
                    4 * track_linear_speed -
                    track_lateral_speed -
                    np.linalg.norm(track_position)
                ) / MAX_SPEED


        if self.reward_type == "time":
            if (track_progress - self.last_rewarded_advance) > PROGRESS_INCREMENT:
                self.last_rewarded_advance = track_progress
                return 1.0 - (time - self.last_lap_time) / (track_progress * REFERENCE_LAP_TIME)
            return 0.0

        return 0.0

    def done_from_telemetry(self, telemetry):
        position = telemetry.position
        velocity = telemetry.velocity

        time = telemetry.time - self.last_reset_time

        track_coordinates = self.track.coordinates(position)
        track_position = self.track.position(track_coordinates)
        track_progress = self.track.progress(track_coordinates)

        # If we're off track for too long, stop
        if np.linalg.norm(track_position) > 1.0:
            if (time - self.last_intrack_time > MAX_OUTTRACK_TIME):
                # print("Reset for outtrack: time={:.2f} progress={:.2f} position={:.2f}".format(
                #     time,
                #     track_progress,
                #     track_position,
                # ))
                return True
        else:
            self.last_intrack_time = time

        # If we stall (STALL_SPEED) for more than MAX_STALL_TIME then stop.
        track_linear_speed = self.track.linear_speed(position, velocity)
        if track_linear_speed > STALL_SPEED:
            self.last_unstall_time = time
        elif (time - self.last_unstall_time > MAX_STALL_TIME):
            # print("Reset for stall: time={:.2f} progress={:.2f} position={:.2f}".format(
            #     time,
            #     track_progress,
            #     track_position,
            # ))
            return True

        # If the last progress is bigger than the current one, it means we just
        # crossed the finish line, stop.
        if self.last_progress > track_progress + 0.02:
            print("LAP TIME: {:.2f}".format(time - self.last_lap_time))
            if self.lap_count == 2:
                # print("Reset for lapcount: time={:.2f}".format(
                #     time,
                # ))
                return True
            self.lap_count += 1
            self.last_lap_time = time
            self.last_rewarded_advance = 0.0
        self.last_progress = track_progress

        return False

    def reset(self):
        """
        `reset` resets the environment to its initial state and returns an
        initial observation (see `observation_from_telemetry`). `reset` must be
        called at least once before calling `step`.
        """
        if self.track_randomized:
            self.track.randomize()

        if self.do_capture:
            self.capture_index += 1
            self.capture = Capture(
                os.path.join(
                    self.capture_set_dir,
                    str(self.worker_index) + '-' + str(self.capture_index),
                ),
                load=False,
            )

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
        self.last_reset_time = telemetry.time

        self.lap_count = 0
        self.last_lap_time = 0.0
        self.last_progress = 0.0
        self.last_unstall_time = 0.0
        self.last_intrack_time = 0.0
        self.last_rewarded_advance = 0.0
        self.last_track_linear_speed = 0.0

        self.camera_stack = None

        observation = self.observation_from_telemetry(telemetry)
        # print("TELEMETRY RESET {}".format(telemetry))

        if self.do_capture:
            self.capture.add_item(
                observation.camera_raw,
                {
                    'time': observation.time,
                    'simulation_track_randomization': self.track.randomization,
                    'simulation_position': observation.position.tolist(),
                    'simulation_velocity': observation.velocity.tolist(),
                    'simulation_angular_velocity': observation.angular_velocity.tolist(),
                    'simulation_track_coordinates': observation.track_coordinates.tolist(),
                    'simulation_track_angle': observation.track_angles[0],
                },
            )

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
        if self.action_type == 'semi_discrete_throttle':
            steering = controls[0]
            throttle= 0.0
            brake = 0.0
            if controls[1] > 0.0:
                throttle = DISCRETE_THROTTLE_SLOW
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

        if self.do_capture:
            self.capture.update_item(
                self.capture.size()-1,
                {
                    'simulation_steering': float(steering),
                    'simulation_throttle': float(throttle),
                    'simulation_brake': float(brake),
                },
            )

        command = Command(steering, throttle, brake)

        self.simulation.step(command)
        telemetry = self.simulation.telemetry()

        observation = self.observation_from_telemetry(telemetry)
        reward = self.reward_from_telemetry(telemetry)
        done = self.done_from_telemetry(telemetry)

        self.step_count += 1

        if not done and self.do_capture:
            self.capture.add_item(
                observation.camera_raw,
                {
                    'time': observation.time,
                    'simulation_track_randomization': self.track.randomization,
                    'simulation_position': observation.position.tolist(),
                    'simulation_velocity': observation.velocity.tolist(),
                    'simulation_angular_velocity': observation.angular_velocity.tolist(),
                    'simulation_track_coordinates': observation.track_coordinates.tolist(),
                    'simulation_track_angle': observation.track_angles[0],
                },
            )

        if done:
            self.reset()
            # If we're done we read the new observations post reset.
            observation = self.observation_from_telemetry(telemetry)

        return observation, reward, done


_send_condition = threading.Condition()
_recv_condition = threading.Condition()
_recv_count = 0

class Worker(threading.Thread):
    def __init__(self, config, index, save_dir=None):
        self.condition = threading.Condition()
        self.controls = None
        self.observation = None
        self.reward = 0.0
        self.done = False
        self.donkey = Donkey(config, save_dir=save_dir, worker_index=index)
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
    def __init__(self, config, save_dir=None):
        self.worker_count = config.get('worker_count')
        self.workers = [
            Worker(config, i, save_dir) for i in range(self.worker_count)
        ]
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
