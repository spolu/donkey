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

from capture import Capture
from track import Track
from utils import str2bool

_capture = None
_track = None
_is_simulation = True

def plot_inputs(start, end, is_simulation=True):
    time = [_capture.get_item(i)['time'] for i in range(start, end)]

    if _is_simulation:
        angular_velocity = [_capture.get_item(i)['simulation_angular_velocity'] for i in range(start, end)]
        acceleration = [_capture.get_item(i)['simulation_acceleration'] for i in range(start, end)]
        throttle = [_capture.get_item(i)['simulation_throttle'] for i in range(start, end)]
        steering = [_capture.get_item(i)['simulation_steering'] for i in range(start, end)]
    else:
        angular_velocity = [_capture.get_item(i)['raspi_imu_angular_velocity'] for i in range(start, end)]
        acceleration = [_capture.get_item(i)['raspi_imu_acceleration'] for i in range(start, end)]
        throttle = [_capture.get_item(i)['raspi_throttle'] for i in range(start, end)]
        steering = [_capture.get_item(i)['raspi_steering'] for i in range(start, end)]
        phone_positions = [_capture.get_item(i)['raspi_phone_position'] for i in range(start, end)]
        orientation = [_capture.get_item(i)['raspi_sensehat_orientation'] for i in range(start, end)]

    plt.figure('acceleration')
    plt.plot(time, acceleration, 'k')
    plt.show()

    plt.figure('angular_velocity')
    plt.plot(time, angular_velocity, 'k')
    plt.show()

    plt.figure('throttle')
    plt.plot(time, throttle, 'k')
    plt.show()

    plt.figure('steering')
    plt.plot(time, steering, 'k')
    plt.show()

    plt.figure('phone position')
    plt.plot(time, phone_positions, 'k')
    plt.show()

    plt.figure('orientation')
    plt.plot(time, orientation, 'k')
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--capture_dir', type=str, help="path to saved captured data")
    parser.add_argument('--track', type=str, help="track name")
    parser.add_argument('--is_simulation', type=str2bool, help="data is from simulation")

    args = parser.parse_args()

    if args.is_simulation is not None:
        _is_simulation = args.is_simulation

    assert args.capture_dir is not None
    _capture = Capture(args.capture_dir, load=True)

    assert args.track is not None
    _track = Track(args.track)

    plot_inputs(0,_capture.__len__())
