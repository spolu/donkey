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

from flask import Flask
from eventlet.green import threading
from utils import Config, str2bool
from capture import Capture
from capture.models import ResNet
from simulation import Donkey
from simulation import ANGLES_WINDOW

_sio = socketio.Server(logging=False, engineio_logger=False)
_app = Flask(__name__)

_d = None

_observations = None
_reward = None
_done = None

# import pdb; pdb.set_trace()

def transition():
    global _observations

    return {
        'done': _done,
        'reward': _reward,
        'progress': _observations.progress,
        'time': _observations.time,
        'linear_speed': _observations.track_linear_speed,
        'camera': _observations.camera_stack[0].tolist(),
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

    torch.manual_seed(cfg.get('seed'))
    random.seed(cfg.get('seed'))
    if cfg.get('cuda'):
        torch.cuda.manual_seed(cfg.get('seed'))

    device = torch.device('cuda:0' if cfg.get('cuda') else 'cpu')
    model = ResNet(cfg, ANGLES_WINDOW+1).to(device)

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

    while True:
        input = torch.tensor(cv2.imdecode(
            np.fromstring(_observations.camera_raw, np.uint8),
            cv2.IMREAD_COLOR,
        ), dtype=torch.float).to(device)

        input = input / 127.5 - 1
        input = input.transpose(0, 2).unsqueeze(0)

        output = model(input)
        print("OUTPUT {:.4f} || {:.4f} {:.4f} {:.4f} {:.4f}".format(
            output[0][-1],
            output[0][0],
            output[0][1],
            output[0][2],
            output[0][3],
        ))

        # steering, throttle_brake = planner.plan(output[:-1], output[-1])

        _observations, _reward, _done = _d.step([0, 1.0])

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

    t = threading.Thread(target = run_server)
    t.start()

    run(cfg)
