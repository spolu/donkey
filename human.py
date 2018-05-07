import time
import socketio
import eventlet
import eventlet.wsgi
import os

from flask import Flask
from eventlet.green import threading
from utils import Config
from simulation import Donkey

sio = socketio.Server(logging=False, engineio_logger=False)
app = Flask(__name__)

d = None
observations = None
reward = None
done = None

def transition():
    return {
        'done': done,
        'reward': reward,
        'progress': observations.progress,
        'time': observations.time,
        'linear_speed': observations.track_linear_speed,
        'camera': observations.camera_stack[0].tolist(),
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
    global observations
    global reward
    global done

    steering = data['steering']
    throttle_brake = 0.0

    if data['brake'] > 0.0:
        throttle_brake = -data['brake']
    if data['throttle'] > 0.0:
        throttle_brake = data['throttle']

    observations, reward, done = d.step([steering, throttle_brake])

    sio.emit('transition', transition())
    sio.emit('next')

@sio.on('reset')
def reset(sid, data):
    d.reset()

if __name__ == "__main__":
    cfg = Config('configs/human.json')

    d = Donkey(cfg)
    observations = d.reset()

    run_server()

