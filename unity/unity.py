import time
import socketio
import sys
import eventlet
import eventlet.wsgi
import collections
import os
import signal
import subprocess
import socket
import atexit

import numpy as np

from flask import Flask
from eventlet.green import threading

from simulation import Telemetry

"""
Shared underlying socket.io server
"""

_sio = socketio.Server(logging=False, engineio_logger=False)
_app = Flask(__name__)
_port = 9093
_inited = False

_clients = []
_lock = threading.Lock()

def client_for_id(client_id):
    global _lock
    global _clients
    with _lock:
        client = _clients[client_id]
    return client

@_sio.on('connect')
def connect(sid, environ):
    # print("Received connect: sid={}".format(sid))
    pass

@_sio.on('disconnect')
def disconnect(sid):
    # print("Received disconnect: sid={}".format(sid))
    pass

@_sio.on('telemetry')
def telemetry(sid, data):
    # print("Received telemetry: sid={} client_id={}".format(sid, data['id']))

    # if int(data['id']) == 0:
    #     print("TIMELOG "
    #           "time={:.3f} fps={:.3f} "
    #           "last_resume={:.3f} last_pause={:.3f} last_telemetry={:.3f} "
    #           "real_delta={:.3f} delta={:.3f} fixed_delta={:.3f} "
    #           "time_scale={:.3f}".format(
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
    if client['callback'] is not None:
        client['callback'](client['telemetry'])


@_sio.on('hello')
def hello(sid, data):
    # print("Received hello: sid={} id={}".format(sid, data['id']))

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
    global _app
    global _sio
    global _port
    # print("Starting shared server: port=" + str(_port))
    address = ('0.0.0.0', _port)
    _app = socketio.Middleware(_sio, _app)
    try:
        eventlet.wsgi.server(eventlet.listen(address), _app)
    except KeyboardInterrupt:
        print("Stopping shared server")

def init_server():
    global _inited
    global _port

    if _inited:
        return

    _port = get_free_tcp_port()

    env = os.environ.copy()
    if 'SIMULATION_PORT' in env:
        _port = int(env['SIMULATION_PORT'])

    # This threading call is imported from eventlet.green. Magic!
    # See http://eventlet.net/doc/patching.html#monkeypatching-the-standard-library
    threading.Thread(target = run_server).start()
    _inited = True

"""
Unity interface
"""

class Unity:
    def __init__(
            self,
            launch,
            headless,
            time_scale,
            step_interval,
            capture_frame_rate,
            callback,
    ):
        global _lock
        global _clients
        self.launch = launch
        self.headless = headless
        self.time_scale = time_scale
        self.step_interval = step_interval
        self.capture_frame_rate = capture_frame_rate
        self.env = os.environ.copy()
        self.process = None

        if 'SIMULATION_PORT' in self.env:
            self.launch = False

        with _lock:
            self.client = {
                'id': len(_clients),
                'condition': threading.Condition(),
                'sid': "",
                'callback': callback,
            }
            _clients.append(self.client)

    def start(self, track):
        global _lock
        global _sio
        global _port

        # Lazily init the shared socket.IO server.
        with _lock:
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
                str(_port),
            ]
            if self.headless:
                cmd.append('-batchmode')

            self.process = subprocess.Popen(cmd, env=self.env)
            self.client['process'] = self.process

        self.client['condition'].wait()

        self.client['condition'].release()
        # print("Unity started: id={} sid={}".format(
        #     self.client['id'], self.client['sid'],
        # ))

        self.client['condition'].acquire()

        with _lock:
            _sio.emit('reset', data={
                'track_path': track.serialize(),
                'track_width': str(track.width()),
            }, room=self.client['sid'])

        self.client['condition'].wait()
        self.client['condition'].release()
        # print("Received initial telemetry: id={} sid={}".format(
        #     self.client['id'], self.client['sid'],
        # ))

    def stop(self):
        if self.launch:
            self.process.terminate()

        # print("Unity stopped: id={} sid={}".format(
        #     self.client['id'], self.client['sid'],
        # ))

    def reset(self, track):
        self.client['condition'].acquire()

        with _lock:
            _sio.emit('reset', data={
                'track_path': track.serialize(),
                'track_width': str(track.width()),
            }, room=self.client['sid'])

        self.client['condition'].wait()
        self.client['condition'].release()
        # print("Received initial telemetry: id={} sid={}".format(
        #     self.client['id'], self.client['sid'],
        # ))

    def step(self, command):
        global _lock
        global _sio

        self.client['condition'].acquire()

        # Send command.
        with _lock:
            _sio.emit('step', data={
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

        return Telemetry(
            telemetry['time'],
            telemetry['camera'],
            np.array([
                telemetry['position']['x'],
                telemetry['position']['y'],
                telemetry['position']['z'],
            ]),
            np.array([
                telemetry['velocity']['x'],
                telemetry['velocity']['y'],
                telemetry['velocity']['z'],
            ]),
            np.array([
                telemetry['angular_velocity']['x'],
                telemetry['angular_velocity']['y'],
                telemetry['angular_velocity']['z'],
            ]),
        )

def cleanup():
    for c in _clients:
        if 'process' in c:
            print("Unity cleaned up: id={} sid={}".format(
                c['id'], ['sid'],
            ))
            c['process'].terminate()

atexit.register(cleanup)
