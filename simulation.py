import threading
import time
import socketio
import eventlet
import eventlet.wsgi
import collections
import os
import subprocess
# import tty, termios, sys

from flask import Flask

sio = socketio.Server()
app = Flask(__name__)
inited = False

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    sio.emit('reset', data={}, sid=sid)

@sio.on('disconnect')
def disconnect(sid):
    # print('disconnect ', sid)
    pass

@sio.on('telemetry')
def telemetry(sid, data):
    # print("telemetry", sid)
    # print("DATA: {}".format(data))
    ##global step
    ### print ("SENDING: {}".format(step))
    sio.emit('step', {}, sid=True)

@sio.on('hello')
def hello(sid, data):
    # print("hello", sid, data['id'])

    client = None
    with lock:
        client = CLIENTS[int(data['id'])]

    client['condition'].acquire()
    client['sid'] = sid
    client['condition'].notify()
    client['condition'].release()

def run_server():
    global app
    print("starting server thread")
    address = ('0.0.0.0', 9090)
    try:
        eventlet.wsgi.server(eventlet.listen(address), app)
    except KeyboardInterrupt:
        print('stopping')

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

    def init(self):
        global lock
        global CLIENTS

        with lock:
            init_server()

        # Start simulation
        cmd = [
            self.env['SIM_PATH'],
        ]
        proc = subprocess.Popen(cmd, env=self.env)

        print("waiting on client condition {}".format(self.client['id']))
        self.client['condition'].acquire()
        self.client['condition'].wait()
        self.client['condition'].release()
        print("Received client notify {} {}".format(
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
    c = Simulation()
    c.init()

if __name__ == "__main__":
    main()
