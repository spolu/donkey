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
from reinforce.algorithms import PPOVAE

_sio = socketio.Server(logging=False, engineio_logger=False)
_app = Flask(__name__)

_observations = None
_reward = None
_done = None
_track = None

def transition():
    return {
        'done': _done,
        'reward': _reward,
        'observation': {
            'track_coordinates': _observations.track_coordinates.tolist(),
            'time': _observations.time,
            'track_linear_speed': _observations.track_linear_speed,
            'camera': _observations.camera.tolist(),
            'position': _observations.position.tolist(),
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
    cfg.override('device', 'cpu')

    if args.capture_set_save_dir != None:
        cfg.override('capture_set_save_dir', args.capture_set_save_dir)
    if args.reinforce_load_dir != None:
        cfg.override('reinforce_load_dir', args.reinforce_load_dir)
    if args.synthetic_load_dir != None:
        cfg.override('synthetic_load_dir', args.synthetic_load_dir)
    if args.unity_headless != None:
        cfg.override('unity_headless', args.unity_headless)
    if args.unity_time_scale != None:
        cfg.override('unity_time_scale', args.unity_time_scale)
    if args.unity_step_interval != None:
        cfg.override('unity_step_interval', args.unity_step_interval)
    if args.unity_capture_frame_rate != None:
        cfg.override('unity_capture_frame_rate', args.unity_capture_frame_rate)

    torch.manual_seed(cfg.get('seed'))
    random.seed(cfg.get('seed'))

    if not args.reinforce_load_dir:
        raise Exception("Required argument: --reinforce_load_dir")

    if cfg.get('algorithm') == 'ppo':
        algorithm = PPO(cfg)
    if cfg.get('algorithm') == 'ppo_vae':
        algorithm = PPOVAE(cfg)
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

    parser.add_argument('--reinforce_load_dir', type=str, help="config override")
    parser.add_argument('--synthetic_load_dir', type=str, help="config override")
    parser.add_argument('--capture_set_save_dir', type=str, help="config override")

    parser.add_argument('--unity_headless', type=str2bool, help="config override")
    parser.add_argument('--unity_time_scale', type=float, help="config override")
    parser.add_argument('--unity_step_interval', type=float, help="config override")
    parser.add_argument('--unity_capture_frame_rate', type=int, help="config override")

    args = parser.parse_args()

    threading.Thread(target = run_server).start()

    run(args)
