import threading
import time
import socketio
import eventlet
import eventlet.wsgi
import collections
import os
import subprocess

from flask import Flask

"""
Shared underlying socket.io server
"""

sio = socketio.Server(logging=True, engineio_logger=True)
app = Flask(__name__)
inited = False

clients = []
lock = threading.Lock()

def client_for_id(client_id):
    global lock
    global clients
    client = None
    with lock:
        client = clients[client_id]
    return client

@sio.on('connect')
def connect(sid, environ):
    print("Received connect: sid={}".format(sid))

@sio.on('disconnect')
def disconnect(sid):
    print("Received disconnect: sid={}".format(sid))

@sio.on('telemetry')
def telemetry(sid, data):
    print("Received telemetry: sid={}".format(sid))

    # Record telemetry on the client and notify.
    client = client_for_id(int(data['id']))
    client['condition'].acquire()
    client['telemetry'] = data
    client['condition'].notify()
    client['condition'].release()

@sio.on('hello')
def hello(sid, data):
    print("Received hello: sid={} id={}".format(sid, data['id']))

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

    app = socketio.Middleware(sio, app)
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
    def __init__(self, launch=True, headless=True):
        global lock
        global clients
        self.headless = headless
        self.launch = launch
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

        # Start simulation.
        if self.launch:
            cmd = [
                self.env['SIM_PATH'],
                "-simulationClientID",
                str(self.client['id']),
            ]
            if self.headless:
                cmd.append('-batchmode')

            self.process = subprocess.Popen(cmd, env=self.env)

        self.client['condition'].acquire()
        self.client['condition'].wait()
        self.client['condition'].release()
        print("Simulation started: id={} sid={}".format(
            self.client['id'], self.client['sid'],
        ))

        with lock:
            sio.emit('reset', data={}, room=self.client['sid'])

        self.client['condition'].acquire()
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

    def step(self, command):
        global lock
        global sio

        # Send command.
        with lock:
            sio.emit('step', data={
                'steering': command.steering,
                'throttle': command.throttle,
                'brake': command.brake,
            }, room=self.client['sid'])

        print(self.client['condition'])

        # Wait for telemetry to be received.
        self.client['condition'].acquire()
        self.client['condition'].wait()
        self.client['condition'].release()

step = {
    "steering": "0.0",
    "throttle": "0.0",
    "brake": "0.0",
}

def main():
    c0 = Simulation(launch=False, headless=False)
    c0.start()

    c0.step(Command(0.0,1.0,0.0))
    print("Command 0")
    c0.step(Command(0.0,1.0,0.0))
    print("Command 1")
    c0.step(Command(0.0,1.0,0.0))
    print("Command 2")
    c0.step(Command(0.0,1.0,0.0))
    print("Command 3")
    time.sleep(2)
    c0.step(Command(0.0,1.0,0.0))
    print("Command 4")
    c0.step(Command(0.0,1.0,0.0))
    print("Command 5")

    # c0.stop()

    # c1 = Simulation(headless=False)
    # c1.start()

if __name__ == "__main__":
    main()
