import simulation
import track
import base64
import cv2
import numpy as np

from eventlet.green import threading

MAX_GAME_TIME = 120
OFF_TRACK_DISTANCE = 6.0
CAMERA_CHANNEL = 3
CAMERA_WIDTH = 120
CAMERA_HEIGHT = 160
CONTROL_SIZE = 2
# Beware on a laptop 4.0 is too fast for the machine and will skip some
# intervals.
SIMULATION_TIME_SCALE = 4.0
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

        speed = self.track.speed(position, velocity)
        distance = self.track.distance(position)

        return speed - distance

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
        - steering is tanh-ed to [-1;1]
        - throttle is tanh-ed to [0;1]
        - brake is tanh-ed to [0;1]
        It returns:
        - the observation as computed by `observation_from_telemetry`
        - a reward value for the last step
        - a boolean indicating whether the game is finished
        """
        steering = np.tanh(controls[0])
        throttle_brake = np.tanh(controls[1])
        if throttle_brake > 0:
            throttle = throttle_brake
            brake = 0
        else:
            throttle = 0.0
            brake = -throttle_brake

        command = simulation.Command(steering, throttle, brake)

        self.simulation.step(command)
        telemetry = self.simulation.telemetry()

        observation = self.observation_from_telemetry(telemetry)
        reward = self.reward_from_telemetry(telemetry)
        done = self.done_from_telemetry(telemetry)

        # if self.step_count % 1000 == 0:
        #     print("TELEMETRY {}".format(telemetry))
        print(">> TIM/POS/VEL/CMD {:.2f} {:.2f} {:.2f} {:.2f} / {:.2f} {:.2f} {:.2f} / {:.2f} {:.2f} {:.2f}".format(
            telemetry['time'],
            telemetry['position']['x'],
            telemetry['position']['y'],
            telemetry['position']['z'],
            telemetry['velocity']['x'],
            telemetry['velocity']['y'],
            telemetry['velocity']['z'],
            steering,
            throttle,
            brake,
        ))
        self.step_count += 1


        if done:
            print(">> DONE")
            self.reset()

        return observation, reward, done


_send_condition = threading.Condition()
_recv_condition = threading.Condition()
_recv_count = 0

class Worker(threading.Thread):
    def __init__(self, headless):
        self.condition = threading.Condition()
        self.controls = None
        self.observation = None
        self.reward = 0.0
        self.done = False
        self.donkey = Donkey(headless=headless)
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

            observation, reward, done = self.donkey.step(self.controls)

            self.observation = observation
            self.reward = reward
            self.done = done

            # Notify that we are done.
            _recv_condition.acquire()
            _recv_count = _recv_count + 1
            _recv_condition.notify_all()
            _recv_condition.release()

class Envs:
    def __init__(self, worker_count, headless):
        self.worker_count = worker_count
        self.workers = [Worker(headless) for _ in range(self.worker_count)]
        for w in self.workers:
            w.start()

    def reset(self):
        for w in self.workers:
            w.reset()
        observations = [w.observation for w in self.workers]

        return np.stack(observations)

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

        return np.stack(observations), np.stack(rewards), np.stack(dones)
