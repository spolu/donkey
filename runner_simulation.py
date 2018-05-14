import sys
import random
import argparse
import signal
import socketio
import eventlet
import eventlet.wsgi
import os.path
import torch

from flask import Flask
from eventlet.green import threading
from utils import Config, str2bool
from capture import Capture
from capture.models import ResNet
from simulation import Donkey

_sio = socketio.Server(logging=False, engineio_logger=False)
_app = Flask(__name__)

_d = None

_observations = None
_reward = None
_done = None

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
    torch.manual_seed(cfg.get('seed'))
    random.seed(cfg.get('seed'))
    if cfg.get('cuda'):
        torch.cuda.manual_seed(cfg.get('seed'))

    device = torch.device('cuda:0' if cfg.get('cuda') else 'cpu')
    model = ResNet(self.config, ANGLES_WINDOW+1).to(device)

    if not args.load_dir:
        raise Exception("Required argument: --load_dir")

    if cfg.get('cuda'):
        model.load_state_dict(
            torch.load(self.load_dir + "/model.pt"),
        )
    else:
        model.load_state_dict(
            torch.load(self.load_dir + "/model.pt", map_location='cpu'),
        )

    while True:
        def step_callback(o, r, d):
            global observations
            global reward
            global done
            global policy

            observations = o[0]
            reward = r
            done = d

            _sio.emit('transition', transition())
            _sio.emit('next')

        reward = model.run(step_callback)

        print("DONE {}".format(reward))

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--load_dir', type=str, help="path to saved models directory")

    parser.add_argument('--simulation_headless', type=str2bool, help="config override")
    parser.add_argument('--simulation_time_scale', type=float, help="config override")
    parser.add_argument('--simulation_step_interval', type=float, help="config override")
    parser.add_argument('--simulation_capture_frame_rate', type=int, help="config override")

    args = parser.parse_args()

    cfg = Config('configs/capture_trainer.json')

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
