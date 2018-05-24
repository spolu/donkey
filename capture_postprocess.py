import time
import socketio
import math
import numpy as np
import base64
import eventlet
import eventlet.wsgi
import os
import collections
import argparse

from capture import Capture
from simulation import Track

NOISE_SAMPLES = 5
NOISE_ACCELERATION_SCALE = 0.1
NOISE_ANGULAR_VELOCITY_SCALE = 0.05

_capture = None
_segments = []

Noise = collections.namedtuple(
    'Noise',
    'index type value'
)

def integrate(noise):
    time = [_capture.get_item(i)['time'] for i in range(_capture.__len__())]
    angular_velocity = [_capture.get_item(i)['angular_velocity'] for i in range(_capture.__len__())]
    acceleration = [_capture.get_item(i)['acceleration'] for i in range(_capture.__len__())]

    if noise != None:
        if noise.type == 'acceleration':
            acceleration[noise.index] += noise.value
        if noise.type == 'angular_velocity':
            angular_velocity[noise.index] += noise.value

    angles = [math.pi]
    for i in range(1, len(time)):
        angles.append(angles[i-1] + angular_velocity[i][1] * (time[i] - time[i-1]))

    velocities = [np.array([0.0, 0.0, 0.0])]
    for i in range(1, len(time)):
        # print("ACCELERATIOn {}".format(acceleration[i]))
        # print("TIME {}".format((time[i] - time[i-1])))
        velocities.append(velocities[i-1] + np.array(acceleration[i]) * (time[i] - time[i-1]))

    positions = [np.array([0,0,0])]
    # for i in range(1, len(time)):
    #     positions.append(
    #         positions[i-1] + [
    #             velocities[i-1][0] * (time[i] - time[i-1]) * 0.8,
    #             velocities[i-1][1] * (time[i] - time[i-1]) * 0.8,
    #             velocities[i-1][2] * (time[i] - time[i-1]) * 0.8,
    #         ],
    #     )
    for i in range(1, len(time)):
        velocities[i-1][1] = 0.0
        speed = np.linalg.norm(velocities[i-1])
        # print("SPEED {}".format(speed))
        positions.append(
            positions[i-1] + [
                -np.sin(angles[i-1]) * speed * (time[i] - time[i-1]),
                0,
                -np.cos(angles[i-1]) * speed * (time[i] - time[i-1]),
            ],
        )

    return angles, velocities, positions

def sample_noises(segment):
    pass

def course_correct(segment):
    pass

def postprocess():
    # Integrate the path and update the _capture.
    print("DIRECT INTEGRATION")

    angles, velocities, positions = integrate(None)
    for i in range(len(positions)):
        progress = _track.progress(positions[i]) / _track.length
        track_position = _track.position(positions[i])
        if np.linalg.norm(velocities[i]) > 0:
            track_angle =  _track.angle(positions[i], velocities[i])
        else:
            track_angle = 0.0
        d = {
            'integrated_progress': progress,
            'integrated_track_position': track_position,
            'integrated_track_angle': track_angle,
        }
        _capture.update_item(i, d, save=True)

    # Course correct segment by segmet and update the _capture.
    # print("DIRECT INTEGRATION")
    # for i in range(len(_segments)):
    #     print("COURSE CORRECTING SEGMENT {}/{} [{},{}]".format(
    #         i, len(_segments), _segments[i][0], _segments[i][1],
    #     ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--capture_dir', type=str, help="path to saved captured data")
    parser.add_argument('--track', type=str, help="track name")

    args = parser.parse_args()

    assert args.capture_dir is not None
    _capture = Capture(args.capture_dir, load=True)

    assert args.track is not None
    _track = Track(args.track)

    # This code assumes that the first point of the path is annotated. It will
    # also only course correct to the last annotated point.
    assert 'annotated_progress' in _capture.get_item(0)

    last = 0
    for i in range(_capture.__len__()):
        if 'annotated_progress' in _capture.get_item(i) and 'annotated_track_position' in _capture.get_item(i):
            if i == 0:
                continue
            _segments.append([last, i])
            last = i

    postprocess()
