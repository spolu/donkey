import threading
import time
import socketio
import eventlet
import eventlet.wsgi
import collections
import os
import subprocess

from flask import Flask

sio = socketio.Server()
app = Flask(__name__)
inited = False

@sio.on('connect')
def connect(sid, environ):
    print("Received connect: sid={}".format(sid))

@sio.on('disconnect')
def disconnect(sid):
    print("Received disconnect: sid={}".format(sid))

@sio.on('telemetry')
def telemetry(sid, data):
    print("Received telemetry: sid={}".format(sid))
    # print("DATA: {}".format(data))
    ##global step
    ### print ("SENDING: {}".format(step))
    # sio.emit('step', {}, sid=True)
    pass

@sio.on('hello')
def hello(sid, data):
    print("Received hello: sid={} id={}".format(sid, data['id']))

    client = None
    with lock:
        client = CLIENTS[int(data['id'])]

    client['condition'].acquire()
    client['sid'] = sid
    client['condition'].notify()
    client['condition'].release()

    sio.emit('reset', data={}, sid=sid)

def run_server():
    global app
    print("Starting socket.IO server: port=9090")
    address = ('0.0.0.0', 9090)
    try:
        eventlet.wsgi.server(eventlet.listen(address), app)
    except KeyboardInterrupt:
        print("Stopping socket.IO server")

def init_server():
    global app
    global sio
    global inited

    if inited:
        return

    app = socketio.Middleware(sio, app)
    threading.Thread(target = run_server).start()
    inited = True

CLIENTS = []
lock = threading.Lock()

class Simulation:
    def __init__(self, headless=True):
        global lock
        global CLIENTS
        self.headless = headless
        self.env = os.environ.copy()

        with lock:
            self.client = {
                'id': len(CLIENTS),
                'condition': threading.Condition(),
                'sid': "",
            }
            CLIENTS.append(self.client)

    def start(self):
        global lock
        global CLIENTS

        # Lazily init the shared socket.IO server.
        with lock:
            init_server()

        # Start simulation.
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

    def stop(self):
        self.process.terminate()
        print("Simulation stopped: id={} sid={}".format(
            self.client['id'], self.client['sid'],
        ))

    def step(self, command):
        # send command and waits for telemetry to be received
        pass

step = {
    "steering": "0.0",
    "throttle": "0.0",
    "brake": "0.0",
}

def main():
    c0 = Simulation(headless=True)
    c0.start()
    # c0.stop()

    # c1 = Simulation(headless=False)
    # c1.start()

if __name__ == "__main__":
    main()
