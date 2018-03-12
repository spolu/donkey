import time
import socketio
import eventlet
import eventlet.wsgi
import os

from flask import Flask
from eventlet.green import threading

import simulation

sio = socketio.Server(logging=False, engineio_logger=False)
app = Flask(__name__)
s = simulation.Simulation(
    True, False, 1.0, 0.2, 0
)

def client_for_sid(sid):
    global lock
    global clients
    with lock:
        client = clients[sid]
    return client

@app.route("/")
def hello():
    return "Donkey"

def run_server():
    global app
    print("Starting shared server: port=9090")
    address = ('0.0.0.0', 9091)
    app = socketio.Middleware(sio, app)
    try:
        eventlet.wsgi.server(eventlet.listen(address), app)
    except KeyboardInterrupt:
        print("Stopping shared server")

@sio.on('connect')
def connect(sid, environ):
    print("Received connect: sid={}".format(sid))
    sio.emit('telemetry', s.telemetry())
    sio.emit('next')

@sio.on('step')
def step(sid, data):
    s.step(
        simulation.Command(
            data['steering'],
            data['throttle'],
            data['brake'],
        ),
    )
    sio.emit('telemetry', s.telemetry())
    sio.emit('next')

@sio.on('reset')
def reset(sid, data):
    s.reset()

if __name__ == "__main__":
    s.start()
    s.reset()

    run_server()

