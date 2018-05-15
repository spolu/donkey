import time
import socketio
import eventlet
import eventlet.wsgi
import collections
import os
import subprocess
import socket

from flask import Flask
from eventlet.green import threading

"""
Shared underlying socket.io server
"""

sio = socketio.Server(logging=False, engineio_logger=False)
app = Flask(__name__)
port = 9093
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

    # if int(data['id']) == 0:
    #     print("TIMELOG time={:.3f} fps={:.3f} last_resume={:.3f} last_pause={:.3f} last_telemetry={:.3f} real_delta={:.3f} delta={:.3f} fixed_delta={:.3f} time_scale={:.3f}".format(
    #         data['time'],
    #         data['fps'],
    #         data['last_resume'],
    #         data['last_pause'],
    #         data['last_telemetry'],
    #         data['last_telemetry'] - data['last_resume'],
    #         data['delta'],
    #         data['fixed_delta'],
    #         data['time_scale'],
    #     ))

    # Record telemetry on the client and notify.
    client = client_for_id(int(data['id']))
    client['condition'].acquire()
    client['telemetry'] = data
    client['condition'].notify()
    client['condition'].release()

@sio.on('hello')
def hello(sid, data):
    # print("Received hello: sid={} id={}".format(sid, data['id']))

    print ("HELLO {}".format(data))

    # Record sid on the client and notify.
    client = client_for_id(int(data['id']))
    client['condition'].acquire()
    client['sid'] = sid
    client['condition'].notify()
    client['condition'].release()

def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port

def run_server():
    global app
    global sio
    global port
    print("Starting shared server: port=" + str(port))
    address = ('0.0.0.0', port)
    app = socketio.Middleware(sio, app)
    try:
        eventlet.wsgi.server(eventlet.listen(address), app)
    except KeyboardInterrupt:
        print("Stopping shared server")

def init_server():
    global inited
    global port

    if inited:
        return

    port = get_free_tcp_port()

    env = os.environ.copy()
    if 'SIMULATION_PORT' in env:
        port = int(env['SIMULATION_PORT'])

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
    def __init__(self, launch, headless, time_scale, step_interval, capture_frame_rate):
        global lock
        global clients
        self.launch = launch
        self.headless = headless
        self.time_scale = time_scale
        self.step_interval = step_interval
        self.capture_frame_rate = capture_frame_rate
        self.env = os.environ.copy()
        self.process = None

        if 'SIMULATION_PORT' in self.env:
            self.launch = False

        with lock:
            self.client = {
                'id': len(clients),
                'condition': threading.Condition(),
                'sid': "",
            }
            clients.append(self.client)

    def start(self, track):
        global lock
        global sio
        global port

        # Lazily init the shared socket.IO server.
        with lock:
            init_server()

        self.client['condition'].acquire()

        sim_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                'build/sim',
            ),
        )
        if os.uname().sysname == 'Darwin':
            sim_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), 
                    'build/sim.app/Contents/MacOS/sim'
                ),
            )

        # Start simulation.
        if self.launch:
            cmd = [
                sim_path,
                "-simulationClientID",
                str(self.client['id']),
                "-simulationTimeScale",
                str(self.time_scale),
                "-simulationStepInterval",
                str(self.step_interval),
                "-simulationCaptureFrameRate",
                str(self.capture_frame_rate),
                "-socketIOPort",
                str(port),
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
            sio.emit('reset', data={
                'track': track.serialize()
            }, room=self.client['sid'])

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

    def reset(self, track):
        self.client['condition'].acquire()

        with lock:
            sio.emit('reset', data={
                'track': track.serialize()
            }, room=self.client['sid'])

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
