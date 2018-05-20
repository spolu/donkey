import time
import socketio
import eventlet
import eventlet.wsgi
import os
import argparse

from flask import Flask
from eventlet.green import threading
from utils import Config, str2bool
from capture import Capture
from simulation import Donkey
from simulation import Track

_sio = socketio.Server(logging=False, engineio_logger=False)
_app = Flask(__name__)

_d = None
_capture = None

_observations = None
_reward = None
_done = None
_track = None

def transition():
    global _observations

    return {
        'done': _done,
        'reward': _reward,
        'observation': {
            'progress': _observations.progress,
            'track_position': _observations.track_position,
            'time': _observations.time,
            'track_linear_speed': _observations.track_linear_speed,
            'camera': _observations.camera_stack[0].tolist(),
            'position': _track.invert_position(
                _observations.progress, _observations.track_position,
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

    _capture.add_item(
        _observations.camera_raw,
        {
            'angular_velocity': _observations.angular_velocity.tolist(),
            'reference_progress': _observations.progress,
            'reference_track_position': _observations.track_position,
            'reference_track_angle': _observations.track_angles[0],
            # For now copy reference to actual values.
            'progress': _observations.progress,
            'track_position': _observations.track_position,
            'track_angle': _observations.track_angles[0],
        },
    )

    _sio.emit('transition', transition())
    _sio.emit('next')

@_sio.on('reset')
def reset(sid, data):
    _d.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--capture_dir', type=str, help="path to saved captured data")

    parser.add_argument('--simulation_headless', type=str2bool, help="config override")
    parser.add_argument('--simulation_time_scale', type=float, help="config override")
    parser.add_argument('--simulation_step_interval', type=float, help="config override")
    parser.add_argument('--simulation_capture_frame_rate', type=int, help="config override")

    args = parser.parse_args()

    cfg = Config('configs/capture_simulation.json')

    if args.simulation_headless != None:
        cfg.override('simulation_headless', args.simulation_headless)
    if args.simulation_time_scale != None:
        cfg.override('simulation_time_scale', args.simulation_time_scale)
    if args.simulation_step_interval != None:
        cfg.override('simulation_step_interval', args.simulation_step_interval)
    if args.simulation_capture_frame_rate != None:
        cfg.override('simulation_capture_frame_rate', args.simulation_capture_frame_rate)

    assert args.capture_dir is not None
    _capture = Capture(args.capture_dir)

    _d = Donkey(cfg)
    _observations = _d.reset()
    _track = Track(cfg.get('track_name'))

    t = threading.Thread(target = run_server)
    t.start()
    t.join()
