import sys
import cv2
import random
import argparse
import signal
import socketio
import eventlet
import eventlet.wsgi
import os.path
import torch

import numpy as np

from flask import Flask
from eventlet.green import threading
from utils import Config, str2bool
from track import Track
from reinforce import Donkey

from reinforce.algorithms import PPO

_sio = socketio.Server(logging=False, engineio_logger=False)
_app = Flask(__name__)

_observations = None
_reward = None
_done = None
_track = None

def transition():
    camera = cv2.imdecode(
        np.fromstring(_observations.camera_raw, np.uint8),
        cv2.IMREAD_GRAYSCALE,
    ).astype(np.float)
    edges = cv2.Canny(
        camera.astype(np.uint8), 50, 150, apertureSize = 3,
    )

    return {
        'done': _done,
        'reward': _reward,
        'observation': {
            'track_coordinates': _observations.track_coordinates.tolist(),
            'time': _observations.time,
            'track_linear_speed': _observations.track_linear_speed,
            'camera': edges.tolist(),
        },
    }

def run_server():
    global _app
    global _sio
    print("Starting shared server: port=9091")
    address = ('0.0.0.0', 9091)
    _app = socketio.Middleware(_sio, _app)
    try:
        eventlet.wsgi.server(eventlet.listen(address), _app)
    except KeyboardInterrupt:
        print("Stopping shared server")

def run(args):
    cfg = Config(args.config_path)

    cfg.override('worker_count', 1)
    cfg.override('cuda', False)

    if args.simulation_headless != None:
        cfg.override('simulation_headless', args.simulation_headless)
    if args.simulation_time_scale != None:
        cfg.override('simulation_time_scale', args.simulation_time_scale)
    if args.simulation_step_interval != None:
        cfg.override('simulation_step_interval', args.simulation_step_interval)
    if args.simulation_capture_frame_rate != None:
        cfg.override('simulation_capture_frame_rate', args.simulation_capture_frame_rate)

    torch.manual_seed(cfg.get('seed'))
    random.seed(cfg.get('seed'))

    if not args.load_dir:
        raise Exception("Required argument: --load_dir")

    if cfg.get('algorithm') == 'ppo':
        algorithm = PPO(cfg, None, args.load_dir)
    assert algorithm is not None

    episode = 0

    while True:
        def step_callback(o, r, d):
            global _observations
            global _reward
            global _done

            _observations = o
            _reward = r
            _done = d

            _sio.emit('transition', transition())
            _sio.emit('next')

        reward = algorithm.test(step_callback)

        print("DONE {}".format(reward))

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('config_path', type=str, help="path to the config file")

    parser.add_argument('--load_dir', type=str, help="path to saved models directory")

    parser.add_argument('--simulation_headless', type=str2bool, help="config override")
    parser.add_argument('--simulation_time_scale', type=float, help="config override")
    parser.add_argument('--simulation_step_interval', type=float, help="config override")
    parser.add_argument('--simulation_capture_frame_rate', type=int, help="config override")

    args = parser.parse_args()

    threading.Thread(target = run_server).start()

    run(args)
