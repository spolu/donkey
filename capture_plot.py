import time
import socketio
import math
import numpy as np
import base64
import eventlet
import random
import eventlet.wsgi
import os
import collections
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from capture import Capture
from track import Track
from utils import str2bool

_capture = None
_track = None
_is_simulation = True
_video_file_path = None

fig, ax = plt.subplots()
positions_x, positions_z = [], []
xdata, ydata = [], []
ln, = plt.plot([], [], "+", animated=True)
time = []


def plot_inputs(start, end, is_simulation=True):
    time = [_capture.get_item(i)['time'] for i in range(start, end)]

    positions_x = []
    positions_z = []
    positions = []
    throttle = []
    steering = []

    if _is_simulation:
        angular_velocity = [_capture.get_item(i)['simulation_angular_velocity'] for i in range(start, end)]
        acceleration = [_capture.get_item(i)['simulation_acceleration'] for i in range(start, end)]
        throttle = [_capture.get_item(i)['simulation_throttle'] for i in range(start, end)]
        steering = [_capture.get_item(i)['simulation_steering'] for i in range(start, end)]
    else:
        throttle = [_capture.get_item(i)['raspi_throttle'] for i in range(start, end)]
        steering = [_capture.get_item(i)['raspi_steering'] for i in range(start, end)]
        for i in range(start, end):
            item = _capture.get_item(i)
            if 'raspi_pozyx_position' not in item:
                continue
            positions.append(item['raspi_pozyx_position'])
            positions_x.append(item['raspi_pozyx_position'][0])
            positions_z.append(item['raspi_pozyx_position'][2])


    print(positions)
    plt.figure('raspi_pozyx_position')
    plt.plot(positions_x, positions_z, 'g^')
    plt.axis('equal')   
    plt.show()

    plt.figure('throttle')
    plt.plot(time, throttle, 'k')
    plt.show()

    plt.figure('steering')
    plt.plot(time, steering, '+')
    plt.show()


def animate_positions(start, end, is_simulation=True):

    for i in range(start, end):
        item = _capture.get_item(i)
        if 'raspi_pozyx_position' not in item:
            continue
        positions_x.append(item['raspi_pozyx_position'][0])
        positions_z.append(item['raspi_pozyx_position'][2])
        time.append(item['time'])

    ani = animation.FuncAnimation(fig, update_animation, frames=np.linspace(0, 500, 500), init_func=init_plot_animation, blit=True, interval=100)
    plt.show()

    if _video_file_path is not None:
        # Set up formatting for the movie files
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=10, metadata=dict(artist='matthieu rouif'), bitrate=1800)
        ani.save(_video_file_path)   


def init_plot_animation():
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal', 'box')
    return ln,

def update_animation(frame):
    print(frame)
    index = int(frame) 
    x = positions_x[index]
    z = positions_z[index]
    xdata.append(x)
    ydata.append(z)
    ln.set_data(xdata, ydata)
    return ln,

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--capture_dir', type=str, help="path to saved captured data")
    parser.add_argument('--track', type=str, help="track name")
    parser.add_argument('--is_simulation', type=str2bool, help="data is from simulation")
    parser.add_argument('--video_file_path', type=str, help="file name")

    args = parser.parse_args()

    if args.is_simulation is not None:
        _is_simulation = args.is_simulation

    assert args.capture_dir is not None
    _capture = Capture(args.capture_dir, load=True)

    assert args.track is not None
    _track = Track(args.track)

    _video_file_path = args.video_file_path
    # plot_inputs(0,_capture.size())
    animate_positions(0,_capture.size())

