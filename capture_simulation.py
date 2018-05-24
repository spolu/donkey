import time
import socketio
import numpy as np
import base64
import eventlet
import eventlet.wsgi
import os
import argparse
import cv2
import simulation

from flask import Flask
from eventlet.green import threading
from utils import Config, str2bool
from capture import Capture

_sio = socketio.Server(logging=False, engineio_logger=False)
_app = Flask(__name__)

_simulation = None
_capture = None
_track = None
_observation = None

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

def process_telemetry(telemetry):
    global _observation
    global _capture

    camera_raw = base64.b64decode(telemetry['camera'])
    camera = cv2.imdecode(
        np.fromstring(camera_raw, np.uint8),
        cv2.IMREAD_GRAYSCALE,
    ).astype(np.float)
    camera = camera / 127.5 - 1

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
    angular_velocity = np.array([
        telemetry['angular_velocity']['x'],
        telemetry['angular_velocity']['y'],
        telemetry['angular_velocity']['z'],
    ])

    time = telemetry['time']
    progress = _track.progress(position) / _track.length
    track_position = _track.position(position)
    track_angle = _track.angle(position, velocity)
    track_linear_speed = _track.linear_speed(position, velocity)

    _capture.add_item(
        camera_raw,
        {
            'time': time,
            'angular_velocity': angular_velocity.tolist(),
            'acceleration': acceleration.tolist(),
            'reference_progress': progress,
            'reference_track_position': track_position,
            'reference_track_angle': track_angle,
        },
    )

    _observation = {
        'progress': progress,
        'track_position': track_position,
        'time': time,
        'track_linear_speed': track_linear_speed,
        'camera': camera.tolist(),
        'position': _track.invert_position(progress, track_position).tolist(),
    }

def transition():
    global _observation

    return {
        'observation': {
            'progress': _observation['progress'],
            'track_position': _observation['track_position'],
            'time': _observation['time'],
            'track_linear_speed': _observation['track_linear_speed'],
            'camera': _observation['camera'],
            'position': _observation['position'],
        },
    }

@_sio.on('connect')
def connect(sid, environ):
    print("Received connect: sid={}".format(sid))
    if _observation is not None:
        _sio.emit('transition', transition())
    _sio.emit('next')

@_sio.on('step')
def step(sid, data):
    global _observation

    steering = data['steering']
    throttle = data['throttle']
    brake = data['brake']

    command = simulation.Command(steering, throttle, brake)
    _simulation.step(command)

    if _observation is not None:
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
    _capture = Capture(args.capture_dir, load=False)

    _simulation = simulation.Simulation(
        True,
        cfg.get('simulation_headless'),
        cfg.get('simulation_time_scale'),
        cfg.get('simulation_step_interval'),
        cfg.get('simulation_capture_frame_rate'),
        process_telemetry,
    )
    _track = simulation.Track(cfg.get('track_name'))
    _simulation.start(_track)

    t = threading.Thread(target = run_server)
    t.start()
    t.join()
