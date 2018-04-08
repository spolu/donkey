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

sio = socketio.Server(logging=False, engineio_logger=False)
app = Flask(__name__)

observations = None
reward = None
done = None
policy = None
gradients = None

def transition():
    return {
        'done': done,
        'reward': reward,
        'progress': observations.progress,
        'time': observations.time,
        'linear_speed': observations.track_linear_speed,
        'camera': observations.camera[0].tolist(),
        'gradients': gradients[0][0].data.numpy().tolist(),
    }

def run_server():
    global app
    print("Starting shared server: port=9090")
    address = ('0.0.0.0', 9091)
    app = socketio.Middleware(sio, app)
    try:
        eventlet.wsgi.server(eventlet.listen(address), app)
    except KeyboardInterrupt:
        print("Stopping shared server")

def run(args):
    global policy

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

    if cfg.get('cuda'):
        torch.cuda.manual_seed(cfg.get('seed'))

    module = __import__('policies.' + cfg.get('policy'))
    policy = getattr(module, cfg.get('policy')).Policy(cfg)

    module = __import__('models.' + cfg.get('model'))
    model = getattr(module, cfg.get('model')).Model(cfg, policy, None, args.load_dir)

    policy.hook_layers()

    episode = 0
    model.initialize()

    while True:
        def step_callback(o, r, d):
            global observations
            global reward
            global done
            global policy
            global gradients

            observations = o[0]
            reward = r
            done = d

            gradients = policy.gradients / policy.gradients.max()

            sio.emit('transition', transition())
            sio.emit('next')
        reward = model.run(step_callback)
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
