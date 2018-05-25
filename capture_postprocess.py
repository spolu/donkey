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
from simulation import Track

NOISE_SAMPLES = 20
NOISE_ANGLE_SCALE = 0.4
NOISE_SPEED_SCALE = 10.0
LOSS_LIMIT = 0.005

_capture = None
_track = None
_segments = []

_start_angle = math.pi
_start_position = np.array([0, 0, 0])
_start_velocity = np.array([0, 0, 0])

Noise = collections.namedtuple(
    'Noise',
    'index type value'
)

def integrate(noises,
              start,
              end,
              start_angle=math.pi,
              start_position=np.array([0,0,0]),
              start_velocity=np.array([0,0,0])):
    time = [_capture.get_item(i)['time'] for i in range(start, end)]
    angular_velocity = [_capture.get_item(i)['angular_velocity'] for i in range(start, end)]
    acceleration = [_capture.get_item(i)['acceleration'] for i in range(start, end)]

    angles = [start_angle]
    for i in range(1, len(time)):
        angles.append(angles[i-1] + angular_velocity[i][1] * (time[i] - time[i-1]))

    velocities = [start_velocity]
    for i in range(1, len(time)):
        velocities.append(velocities[i-1] + np.array(acceleration[i]) * (time[i] - time[i-1]))

    speeds = []
    for i in range(len(velocities)):
        velocities[i-1][1] = 0.0
        speeds.append(np.linalg.norm(velocities[i-1]))

    for n in noises:
        if n.type == 'angle':
            angles[n.index] += n.value
        if n.type == 'speed':
            speeds[n.index] += n.value

    positions = [start_position]

    for i in range(1, len(time)):
        positions.append(
            positions[i-1] + [
                -np.sin(angles[i-1]) * speeds[i-1] * (time[i] - time[i-1]),
                0,
                -np.cos(angles[i-1]) * speeds[i-1] * (time[i] - time[i-1]),
            ],
        )

    return angles, velocities, positions

def sample_noises(segment):
    noises = []
    for i in range(NOISE_SAMPLES):
        noises.append(Noise(
            random.randrange(_segments[segment][0], _segments[segment][1]) - _segments[segment][0],
            'angle',
            np.random.normal(0, NOISE_ANGLE_SCALE),
        ))
        noises.append(Noise(
            random.randrange(_segments[segment][0], _segments[segment][1]) - _segments[segment][0],
            'speed',
            np.random.normal(0, NOISE_SPEED_SCALE),
        ))
    return noises

def loss(segment, progress, track_position):
    annotated_progress = _capture.get_item(_segments[segment][1])['annotated_progress']
    progress_loss = math.fabs(progress - annotated_progress)

    annotated_track_position = _capture.get_item(_segments[segment][1])['annotated_track_position']
    track_position_loss = math.fabs(track_position - annotated_track_position)

    return max(progress_loss, track_position_loss)

def course_correct(segment):
    global _start_angle
    global _start_position
    global _start_velocity

    last_loss = loss(
        segment,
        _capture.get_item(_segments[segment][1])['corrected_progress'],
        _capture.get_item(_segments[segment][1])['corrected_track_position'],
    )
    print("INITIAL LOSS: {:.4f}".format(last_loss))
    noises = []
    running = last_loss > LOSS_LIMIT

    while(running):
        sample = sample_noises(segment)
        losses = []

        for n in sample:
            test = noises + [n]
            angles, velocities, positions = integrate(
                test,
                _segments[segment][0],
                _segments[segment][1] + 1,
                start_angle=_start_angle,
                start_position=_start_position,
                start_velocity=_start_velocity,
            )

            last = _segments[segment][1] - _segments[segment][0]
            progress = _track.progress(positions[last]) / _track.length
            track_position = _track.position(positions[last])

            losses.append(loss(segment, progress, track_position))

        idx = np.argmin(losses)
        l = losses[idx]
        print("LOSS {:.4f} {:.4f}".format(l, last_loss))

        if l > last_loss:
            continue

        last_loss = l
        noises.append(sample[idx])

        if l < LOSS_LIMIT:
            running = False

    # Reintegrate to compute the new start conditions and store the corrected
    # values.
    angles, velocities, positions = integrate(
        noises,
        _segments[segment][0],
        _capture.__len__(),
        # _segments[segment][0],
        # _segments[segment][1] + 1,
        start_angle=_start_angle,
        start_position=_start_position,
        start_velocity=_start_velocity,

    )
    for i in range(len(positions)):
        progress = _track.progress(positions[i]) / _track.length
        track_position = _track.position(positions[i])
        if np.linalg.norm(velocities[i]) > 0:
            track_angle =  _track.angle(positions[i], velocities[i])
        else:
            track_angle = 0.0
        d = {
            'corrected_progress': progress,
            'corrected_track_position': track_position,
            'corrected_track_angle': track_angle,
        }
        _capture.update_item(_segments[segment][0] + i, d, save=False)

    final_loss = loss(
        segment,
        _capture.get_item(_segments[segment][1])['corrected_progress'],
        _capture.get_item(_segments[segment][1])['corrected_track_position'],
    )
    print("FINAL LOSS {:.4f}".format(final_loss))

    last = _segments[segment][1] - _segments[segment][0]

    _start_angle = angles[last]
    _start_position = positions[last]
    _start_velocity = velocities[last]


def postprocess():
    # Integrate the path and update the _capture.
    print("Starting direct integration...")

    angles, velocities, positions = integrate([], 0, _capture.__len__())
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
            'corrected_progress': progress,
            'corrected_track_position': track_position,
            'corrected_track_angle': track_angle,
        }
        _capture.update_item(i, d, save=False)

    # Course correct segment by segmet and update the _capture.
    print("Starting course correction...")
    for s in range(len(_segments)):
        print("Processing segment {}/{} [{},{}]".format(
            s, len(_segments), _segments[s][0], _segments[s][1],
        ))
        course_correct(s)

    for s in range(len(_segments)):
        final_loss = loss(
            s,
            _capture.get_item(_segments[s][1])['corrected_progress'],
            _capture.get_item(_segments[s][1])['corrected_track_position'],
        )
        print("FINAL LOSS {:.4f}".format(final_loss))

    # Saving capture.
    print("Saving capture...")
    for i in range(_capture.__len__()):
        _capture.update_item(i, {}, save=True)

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
