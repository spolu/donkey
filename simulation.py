import time
import socketio
import eventlet
import eventlet.wsgi
import collections
import os
import subprocess

from flask import Flask
from eventlet.green import threading

"""
Shared underlying socket.io server
"""

sio = socketio.Server(logging=False, engineio_logger=False)
app = Flask(__name__)
inited = False

clients = []
lock = threading.Lock()

def client_for_id(client_id):
    global lock
    global clients
    with lock:
        client = clients[client_id]
    return client

@sio.on('connect')
def connect(sid, environ):
    # print("Received connect: sid={}".format(sid))
    pass

@sio.on('disconnect')
def disconnect(sid):
    # print("Received disconnect: sid={}".format(sid))
    pass

@sio.on('telemetry')
def telemetry(sid, data):
    # print("Received telemetry: sid={} client_id={}".format(sid, data['id']))

    # Record telemetry on the client and notify.
    client = client_for_id(int(data['id']))
    client['condition'].acquire()
    client['telemetry'] = data
    client['condition'].notify()
    client['condition'].release()

@sio.on('hello')
def hello(sid, data):
    # print("Received hello: sid={} id={}".format(sid, data['id']))

    # Record sid on the client and notify.
    client = client_for_id(int(data['id']))
    client['condition'].acquire()
    client['sid'] = sid
    client['condition'].notify()
    client['condition'].release()

def run_server():
    global app
    print("Starting shared server: port=9090")
    address = ('0.0.0.0', 9090)
    app = socketio.Middleware(sio, app)
    try:
        eventlet.wsgi.server(eventlet.listen(address), app)
    except KeyboardInterrupt:
        print("Stopping shared server")

def init_server():
    global app
    global sio
    global inited

    if inited:
        return

    # This threading call is imported from eventlet.green. Magic!
    # See http://eventlet.net/doc/patching.html#monkeypatching-the-standard-library
    threading.Thread(target = run_server).start()
    inited = True

"""
Simulation interface
"""

Command = collections.namedtuple(
    'Command',
    'steering throttle brake',
)

class Simulation:
    def __init__(self, launch, headless, time_scale, step_interval):
        global lock
        global clients
        self.headless = headless
        self.launch = launch
        self.time_scale = time_scale
        self.step_interval = step_interval
        self.env = os.environ.copy()
        self.process = None

        with lock:
            self.client = {
                'id': len(clients),
                'condition': threading.Condition(),
                'sid': "",
            }
            clients.append(self.client)

    def start(self):
        global lock
        global sio

        # Lazily init the shared socket.IO server.
        with lock:
            init_server()

        self.client['condition'].acquire()

        # Start simulation.
        if self.launch:
            cmd = [
                self.env['SIM_PATH'],
                "-simulationClientID",
                str(self.client['id']),
                "-simulationTimeScale",
                str(self.time_scale),
                "-simulationStepInterval",
                str(self.step_interval),
            ]
            if self.headless:
                cmd.append('-batchmode')

            print(cmd)

            self.process = subprocess.Popen(cmd, env=self.env)

        self.client['condition'].wait()
        self.client['condition'].release()
        print("Simulation started: id={} sid={}".format(
            self.client['id'], self.client['sid'],
        ))

        self.client['condition'].acquire()

        with lock:
            sio.emit('reset', data={}, room=self.client['sid'])

        self.client['condition'].wait()
        self.client['condition'].release()
        print("Received initial telemetry: id={} sid={}".format(
            self.client['id'], self.client['sid'],
        ))

    def stop(self):
        if self.launch:
            self.process.terminate()

        print("Simulation stopped: id={} sid={}".format(
            self.client['id'], self.client['sid'],
        ))

    def reset(self):
        self.client['condition'].acquire()

        with lock:
            sio.emit('reset', data={}, room=self.client['sid'])

        self.client['condition'].wait()
        self.client['condition'].release()
        # print("Received initial telemetry: id={} sid={}".format(
        #     self.client['id'], self.client['sid'],
        # ))

    def step(self, command):
        global lock
        global sio

        self.client['condition'].acquire()

        # Send command.
        with lock:
            sio.emit('step', data={
                'steering': str(command.steering),
                'throttle': str(command.throttle),
                'brake': str(command.brake),
            }, room=self.client['sid'])

        # Wait for telemetry to be received.
        self.client['condition'].wait()
        self.client['condition'].release()

    def telemetry(self):
        self.client['condition'].acquire()
        telemetry = self.client['telemetry']
        self.client['condition'].release()
        return telemetry
