import sys
import random
import argparse
import signal
import socketio
import eventlet
import eventlet.wsgi
import os.path
import torch
import numpy as np
import cv2

import capture
from capture.models import ResNet
import planner
from planner import Planner

from flask import Flask
from eventlet.green import threading
from utils import Config, str2bool
from reinforce import Donkey
from track import Track

_sio = socketio.Server(logging=False, engineio_logger=False)
_app = Flask(__name__)

_d = None

_observations = None
_reward = None
_done = None
_track = None
_model = {
    'track_progress': 0.0,
    'track_position': 0.0,
    'track_angle': 0.0,
}

# import pdb; pdb.set_trace()

def transition():
    global _observations

    return {
        'done': _done,
        'reward': _reward,
        'observation': {
            'track_progress': _observations.track_progress,
            'track_position': _observations.track_position,
            'time': _observations.time,
            'track_linear_speed': _observations.track_linear_speed,
            'camera': _observations.camera_stack[0].tolist(),
            'position': _track.invert_position(
                _observations.track_progress, _observations.track_position,
            ).tolist(),
        },
        'model': {
            'track_progress': _model['track_progress'],
            'track_position': _model['track_position'],
            'track_angle': _model['track_angle'],
            'position': _track.invert_position(
                _model['track_progress'],
                _model['track_position'],
            ).tolist(),
        }
    }

def run_server():
    global _app
    global _sio

    print("Starting shared server: port=9090")
    address = ('0.0.0.0', 9091)
    _app = socketio.Middleware(_sio, _app)
    try:
        eventlet.wsgi.server(eventlet.listen(address), _app)
    except KeyboardInterrupt:
        print("Stopping shared server")

def run(cfg):
    global _observations
    global _reward
    global _done
    global _model

    torch.manual_seed(cfg.get('seed'))
    random.seed(cfg.get('seed'))
    if cfg.get('cuda'):
        torch.cuda.manual_seed(cfg.get('seed'))

    device = torch.device('cuda:0' if cfg.get('cuda') else 'cpu')
    model = ResNet(cfg, 3).to(device)

    if not args.load_dir:
        raise Exception("Required argument: --load_dir")

    if cfg.get('cuda'):
        model.load_state_dict(
            torch.load(args.load_dir + "/model.pt"),
        )
    else:
        model.load_state_dict(
            torch.load(args.load_dir + "/model.pt", map_location='cpu'),
        )

    model.eval()

    planner = Planner(cfg)

    while True:
        output = model(capture.input_from_camera(
            _observations.camera_raw, device,
        ).unsqueeze(0))

        track_progress = output[0][0].item()
        track_position = output[0][1].item()
        track_angle = output[0][2].item()

        _model = {
            'track_progress': track_progress,
            'track_position': track_position,
            'track_angle': track_angle,
        }

        print("OUTPUT     {:.4f} {:.4f} {:.4f}".format(
            track_progress,
            track_position,
            track_angle,
        ))
        print("SIMULATION {:.4f} {:.4f} {:.4f}".format(
            _observations.track_progress,
            _observations.track_position,
            _observations.track_angles[0],
        ))

        steering, throttle_brake = planner.plan(
            track_progress, track_position, track_angle,
        )

        _observations, _reward, _done = _d.step([steering, throttle_brake])

        _sio.emit('transition', transition())
        _sio.emit('next')

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--load_dir', type=str, help="path to saved models directory")

    parser.add_argument('--simulation_headless', type=str2bool, help="config override")
    parser.add_argument('--simulation_time_scale', type=float, help="config override")
    parser.add_argument('--simulation_step_interval', type=float, help="config override")
    parser.add_argument('--simulation_capture_frame_rate', type=int, help="config override")

    args = parser.parse_args()

    cfg = Config('configs/runner_simulation.json')

    if args.simulation_headless != None:
        cfg.override('simulation_headless', args.simulation_headless)
    if args.simulation_time_scale != None:
        cfg.override('simulation_time_scale', args.simulation_time_scale)
    if args.simulation_step_interval != None:
        cfg.override('simulation_step_interval', args.simulation_step_interval)
    if args.simulation_capture_frame_rate != None:
        cfg.override('simulation_capture_frame_rate', args.simulation_capture_frame_rate)

    _d = Donkey(cfg)
    _observations = _d.reset()
    _track = Track(cfg.get('track_name'))

    t = threading.Thread(target = run_server)
    t.start()

    run(cfg)
