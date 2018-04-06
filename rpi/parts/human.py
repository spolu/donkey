import time
import socketio
import eventlet
import eventlet.wsgi
import os

from flask import Flask

sio = socketio.Server(logging=False, engineio_logger=False)
app = Flask(__name__)

camera = None
throttle = 0.0
steering = 0.0

def transition():
    return {
        'done': None,
        'reward': None,
        'progress': None,
        'time': None,
        'linear_speed': None,
        'camera': None,
    }

@sio.on('connect')
def connect(sid, environ):
    print("Received connect: sid={}".format(sid))
    sio.emit('transition', transition())
    sio.emit('next')

@sio.on('step')
def step(sid, data):
    global throttle
    global steering

    steering = data['steering']
    throtle = data['throttle']

    sio.emit('transition', transition())
    sio.emit('next')

class Human:
    def __init__(self):
        print('Human loaded')

    def update(self, port=9092):
        global app
        print("Starting shared server: port={}".format(port))
        address = ('0.0.0.0', port)
        app = socketio.Middleware(sio, app)
        eventlet.wsgi.server(eventlet.listen(address), app)

    def run_threaded(self, img_arr=None):
        global camera
        global steering
        global throttle

        print("RUN HUMAN")
        print(img_arr)

        camera = img_arr
        return steering, throttle

    def run(self, img_arr=None):
        raise Exception("We expect for this part to be run with the threaded=True argument.")
        return False

    def shutdown(self):
        print('stoping Human')
