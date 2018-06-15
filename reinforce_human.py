import time
import socketio
import eventlet
import eventlet.wsgi
import os
import simulation

from flask import Flask
from eventlet.green import threading
from utils import Config
from track import Track
from reinforce import Donkey

sio = socketio.Server(logging=False, engineio_logger=False)
app = Flask(__name__)

d = None
_observations = None
_reward = None
_done = None
_track = None

def transition():
    return {
        'done': _done,
        'reward': _reward,
        'observation': {
            'track_progress': _observations.track_progress.tolist(),
            'track_position': _observations.track_position,
            'time': _observations.time,
            'track_linear_speed': _observations.track_linear_speed,
            'camera': _observations.camera_stack[0].tolist(),
            'position': _track.invert(
                _observations.track_progress, _observations.track_position,
            ).tolist(),
        },
    }

def run_server():
    global app
    print("Starting shared server: port=9091")
    address = ('0.0.0.0', 9091)
    app = socketio.Middleware(sio, app)
    try:
        eventlet.wsgi.server(eventlet.listen(address), app)
    except KeyboardInterrupt:
        print("Stopping shared server")

@sio.on('connect')
def connect(sid, environ):
    print("Received connect: sid={}".format(sid))
    sio.emit('transition', transition())
    sio.emit('next')

@sio.on('step')
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

    _observations, _reward, _done = d.step([steering, throttle_brake])

    sio.emit('transition', transition())
    sio.emit('next')

@sio.on('reset')
def reset(sid, data):
    d.reset()

if __name__ == "__main__":
    cfg = Config('configs/human.json')

    d = Donkey(cfg)
    _observations = d.reset()
    _track = Track(cfg.get('track_name'))

    run_server()

