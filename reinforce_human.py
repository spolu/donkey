import time
import cv2
import socketio
import argparse
import eventlet
import eventlet.wsgi
import os
import simulation

import numpy as np

from flask import Flask
from eventlet.green import threading
from utils import Config, str2bool
from track import Track

import reinforce

_sio = socketio.Server(logging=False, engineio_logger=False)
_app = Flask(__name__)

_d = None
_observations = None
_reward = None
_done = None
_track = None

def transition():
    camera = cv2.imdecode(
        np.fromstring(_observations.camera_raw, np.uint8),
        cv2.IMREAD_GRAYSCALE,
    ).astype(np.float)
    # camera = camera / 127.5 - 1
    edges = cv2.Canny(
        cv2.resize(
            camera.astype(np.uint8),
            (
                int(reinforce.CAMERA_WIDTH/2),
                int(reinforce.CAMERA_HEIGHT/2),
            ),
            interpolation=cv2.INTER_CUBIC,
        ),
        50, 150, apertureSize = 3,
    )
    edges = edges / 127.5 - 1

    return {
        'done': _done,
        'reward': _reward,
        'observation': {
            'track_coordinates': _observations.track_coordinates.tolist(),
            'time': _observations.time,
            'track_linear_speed': _observations.track_linear_speed,
            'camera': edges.tolist(),
            'position': _track.invert(
                _observations.track_coordinates,
            ).tolist(),
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

@_sio.on('connect')
def connect(sid, environ):
    print("Received connect: sid={}".format(sid))
    _sio.emit('transition', transition())
    _sio.emit('next')

@_sio.on('step')
def step(sid, data):
    global _observations
    global _reward
    global _done

    steering = data['steering']
    throttle_brake = 0.0

    if data['brake'] > 0.0:
        throttle_brake = -data['brake']
    if data['throttle'] > 0.0:
        throttle_brake = data['throttle']

    _observations, _reward, _done = _d.step([steering, throttle_brake])

    _sio.emit('transition', transition())
    _sio.emit('next')

@_sio.on('reset')
def reset(sid, data):
    _d.reset()
    _sio.emit('next')

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--simulation_headless', type=str2bool, help="config override")
    parser.add_argument('--simulation_time_scale', type=float, help="config override")
    parser.add_argument('--simulation_step_interval', type=float, help="config override")
    parser.add_argument('--simulation_capture_frame_rate', type=int, help="config override")

    args = parser.parse_args()

    cfg = Config('configs/human.json')

    if args.simulation_headless != None:
        cfg.override('simulation_headless', args.simulation_headless)
    if args.simulation_time_scale != None:
        cfg.override('simulation_time_scale', args.simulation_time_scale)
    if args.simulation_step_interval != None:
        cfg.override('simulation_step_interval', args.simulation_step_interval)
    if args.simulation_capture_frame_rate != None:
        cfg.override('simulation_capture_frame_rate', args.simulation_capture_frame_rate)

    _d = reinforce.Donkey(cfg)
    _observations = _d.reset()
    _track = Track(cfg.get('track_name'))

    run_server()

