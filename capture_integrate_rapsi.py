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

from utils import str2bool
from capture import Capture
from track import Track

_capture = None
_track = None

_start_angle = math.pi
_start_position = np.array([0, 0, 0])
_start_velocity = np.array([0, 0, 0])

FIXED_SPEED = 1.5

def integrate(start,
              end,
              start_angle=math.pi,
              start_position=np.array([0,0,0]),
              start_speed=FIXED_SPEED):
    time = [_capture.get_item(i)['time'] for i in range(start, end)]
    orientation = [_capture.get_item(i)['raspi_sensehat_orientation'] for i in range(start, end)]
    angular_velocity = [_capture.get_item(i)['raspi_imu_angular_velocity'] for i in range(start, end)]
    acceleration = [_capture.get_item(i)['raspi_imu_acceleration'] for i in range(start, end)]

    # angles = [start_angle]
    # for i in range(1, len(time)):
    #     angles.append(orientation[i][1] - orientation[0][1] + start_angle)

    angles = [start_angle]
    for i in range(1, len(time)):
        # print("ANGULAR_VELOCITY: {}".format(angular_velocity[i][1]))
        angles.append(angles[i-1] - angular_velocity[i][1] / 57.2958 * (time[i] - time[i-1]))

    speeds = [start_speed]
    for i in range(1, len(time)):
        # print("{}".format(acceleration[i][0]))
        # speeds.append(speeds[i-1] + acceleration[i][0] * (time[i] - time[i-1]))
        speeds.append(start_speed)

    positions = [start_position]
    for i in range(1, len(time)):
        positions.append(
            positions[i-1] + [
                -np.sin(angles[i-1]) * speeds[i-1] * (time[i] - time[i-1]),
                0,
                -np.cos(angles[i-1]) * speeds[i-1] * (time[i] - time[i-1]),
            ],
        )
        # print("POSITION: {}".format(positions[i-1]))

    # for i in range(1, len(angles)):
    #     print("{},{},{},{},{}".format(
    #         time[i-1],
    #         time[i],
    #         orientation[i][0],
    #         orientation[i][1],
    #         orientation[i][2],
    #     ))
    # for i in range(1, len(angles)):
    #     print("{},{},{},{},{}".format(
    #         time[i-1],
    #         time[i],
    #         angular_velocity[i][0],
    #         angular_velocity[i][1],
    #         angular_velocity[i][2],
    #     ))


    return angles, positions

def integrate_raspi():
    # Integrate the path and update the _capture.
    print("Starting direct integration...")

    angles, positions = integrate(0, _capture.__len__())
    for i in range(len(positions)):
        track_progress = _track.progress(positions[i])
        track_position = _track.position(positions[i])
        d = {
            'integrated_track_progress': track_progress,
            'integrated_track_position': track_position,
            'corrected_track_progress': track_progress,
            'corrected_track_position': track_position,
        }
        _capture.update_item(i, d, save=False)

    # Saving capture.
    print("Saving capture...")
    _capture.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--capture_dir', type=str, help="path to saved captured data")
    parser.add_argument('--track', type=str, help="track name")

    args = parser.parse_args()

    assert args.capture_dir is not None
    _capture = Capture(args.capture_dir, load=True)

    assert args.track is not None
    _track = Track(args.track)

    integrate_raspi()
