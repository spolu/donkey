import simulation
import track
import base64
import cv2
import numpy as np

from eventlet.green import threading

MAX_GAME_TIME = 30
OFF_TRACK_DISTANCE = 6.0
CAMERA_CHANNEL = 3
CAMERA_WIDTH = 120
CAMERA_HEIGHT = 160
CONTROL_SIZE = 2
MIN_REWARD_SPEED = 0.5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Donkey:
    def __init__(self, config):
        self.simulation_headless = config.get('simulation_headless')
        self.simulation_time_scale = config.get('simulation_time_scale')
        self.simulation_step_interval = config.get('simulation_step_interval')
        self.started = False
        self.simulation = simulation.Simulation(
            True,
            self.simulation_headless,
            self.simulation_time_scale,
            self.simulation_step_interval,
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

        unity = self.track.unity(position)

        return unity, position, velocity, acceleration, camera

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

        # print("SPEED {}".format(speed))

        if speed > MIN_REWARD_SPEED:
            return speed * (1 - distance / OFF_TRACK_DISTANCE)
        else:
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
        `step` takes as input a 2D float numpy array.
        It returns:
        - the observation as computed by `observation_from_telemetry`
        - a reward value for the last step
        - a boolean indicating whether the game is finished
        """
        steering = 2 * sigmoid(controls[0]) - 1.0
        throttle_brake = 2 * sigmoid(controls[1]) - 0.6
        if throttle_brake > 0:
            throttle = throttle_brake
            brake = 0
        else:
            throttle = 0.0
            brake = -throttle_brake

        # print("STEERING {} {}".format(steering, controls[0]))
        # print("THROTTLE {} {}".format(throttle, controls[1]))

        command = simulation.Command(steering, throttle, brake)

        self.simulation.step(command)
        telemetry = self.simulation.telemetry()

        observation = self.observation_from_telemetry(telemetry)
        reward = self.reward_from_telemetry(telemetry)
        done = self.done_from_telemetry(telemetry)

        # if self.step_count % 10 == 0:
        #     print("TELEMETRY {}".format(telemetry))
        # print(">> TIM/POS/VEL/CMD {:.2f} {:.2f} {:.2f} {:.2f} / {:.2f} {:.2f} {:.2f} / {:.2f} {:.2f} {:.2f}".format(
        #     telemetry['time'],
        #     telemetry['position']['x'],
        #     telemetry['position']['y'],
        #     telemetry['position']['z'],
        #     telemetry['velocity']['x'],
        #     telemetry['velocity']['y'],
        #     telemetry['velocity']['z'],
        #     steering,
        #     throttle,
        #     brake,
        # ))
        self.step_count += 1


        if done:
            # print(">> DONE")
            self.reset()

        # print("REWARD: {}".format(reward))

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
    def __init__(self, config):
        self.worker_count = config.get('worker_count')
        self.workers = [Worker(config) for _ in range(self.worker_count)]
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
