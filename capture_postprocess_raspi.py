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

NOISE_SAMPLES = 20

NOISE_ANGLE_SCALE = 0.2
NOISE_SPEED_SCALE = 1.0

LOSS_LIMIT = 0.01

_capture = None
_track = None
_segments = []
_speed = None
_max = None

_start_angle = math.pi
_start_position = np.array([0, 0, 0])
_start_speed = None

Noise = collections.namedtuple(
    'Noise',
    'index type value'
)

def integrate(
        noises,
        start,
        end,
        start_angle,
        start_position,
        start_speed,
):
    time = [_capture.get_item(i)['time'] for i in range(start, end)]
    # orientation = [_capture.get_item(i)['raspi_sensehat_orientation'] for i in range(start, end)]
    angular_velocity = [_capture.get_item(i)['raspi_imu_angular_velocity'] for i in range(start, end)]
    acceleration = [_capture.get_item(i)['raspi_imu_acceleration'] for i in range(start, end)]

    angles = [start_angle]
    for i in range(1, len(time)):
        # print("ANGULAR_VELOCITY: {}".format(angular_velocity[i][1]))
        angles.append(angles[i-1] - angular_velocity[i][1] / 57.2958 * (time[i] - time[i-1]))

    speeds = [start_speed]
    for i in range(1, len(time)):
        # print("{}".format(acceleration[i][0]))
        # speeds.append(speeds[i-1] + acceleration[i][0] * (time[i] - time[i-1]))
        speeds.append(start_speed)

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

    return angles, speeds, positions

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

def loss(segment, track_progress, track_position):
    annotated_track_progress = _capture.get_item(_segments[segment][1])['annotated_track_progress']
    track_progress_loss = math.fabs(track_progress - annotated_track_progress)

    annotated_track_position = _capture.get_item(_segments[segment][1])['annotated_track_position']
    track_position_loss = math.fabs(track_position - annotated_track_position)

    return max(track_progress_loss, track_position_loss)

def course_correct(segment):
    global _start_angle
    global _start_position
    global _start_speed

    _start_speed = _speed

    last_loss = loss(
        segment,
        _capture.get_item(_segments[segment][1])['corrected_track_progress'],
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
            angles, speeds, positions = integrate(
                test,
                _segments[segment][0],
                _segments[segment][1] + 1,
                start_angle=_start_angle,
                start_position=_start_position,
                start_speed=_start_speed,
            )

            last = _segments[segment][1] - _segments[segment][0]
            track_progress = _track.progress(positions[last])
            track_position = _track.position(positions[last])

            losses.append(loss(segment, track_progress, track_position))

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
    angles, speeds, positions = integrate(
        noises,
        _segments[segment][0],
        _capture.__len__(),
        # _segments[segment][0],
        # _segments[segment][1] + 1,
        start_angle=_start_angle,
        start_position=_start_position,
        start_speed=_start_speed,

    )
    for i in range(len(positions)):
        track_progress = _track.progress(positions[i])
        track_position = _track.position(positions[i])
        d = {
            'corrected_track_progress': track_progress,
            'corrected_track_position': track_position,
        }
        _capture.update_item(_segments[segment][0] + i, d, save=False)

    final_loss = loss(
        segment,
        _capture.get_item(_segments[segment][1])['corrected_track_progress'],
        _capture.get_item(_segments[segment][1])['corrected_track_position'],
    )
    print("FINAL LOSS {:.4f}".format(final_loss))

    last = _segments[segment][1] - _segments[segment][0]

    _start_angle = angles[last]
    _start_position = positions[last]
    _start_speed = speeds[last]

def postprocess():
    # Course correct segment by segmet and update the _capture.
    print("Starting course correction...")

    # Reinitialize first corrected value to the integrated value.
    _capture.update_item(_segments[0][1], {
        'corrected_track_progress': _capture.get_item(_segments[0][1])['integrated_track_progress'],
        'corrected_track_position': _capture.get_item(_segments[0][1])['integrated_track_position'],
    }, save=False)

    for s in range(len(_segments)):
        if s >= _max:
            break
        print("Processing segment {}/{} [{},{}]".format(
            s, len(_segments), _segments[s][0], _segments[s][1],
        ))
        course_correct(s)

    for s in range(len(_segments)):
        final_loss = loss(
            s,
            _capture.get_item(_segments[s][1])['corrected_track_progress'],
            _capture.get_item(_segments[s][1])['corrected_track_position'],
        )
        print("FINAL LOSS {:.4f}".format(final_loss))

    # Saving capture.
    print("Saving capture...")
    _capture.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--capture_dir', type=str, help="path to saved captured data")
    parser.add_argument('--track', type=str, help="track name")
    parser.add_argument('--speed', type=float, help="fixed speed")
    parser.add_argument('--max', type=float, help="course correction max")

    args = parser.parse_args()

    assert args.capture_dir is not None
    _capture = Capture(args.capture_dir, load=True)

    assert args.track is not None
    _track = Track(args.track)

    assert args.speed is not None
    _speed = args.speed

    if args.max is not None:
        _max = args.max

    # This code assumes that the first point of the path is annotated. It will
    # also only course correct to the last annotated point.
    assert 'annotated_track_progress' in _capture.get_item(0)

    last = 0
    for i in range(_capture.__len__()):
        if ('annotated_track_progress' in _capture.get_item(i) and
                'annotated_track_position' in _capture.get_item(i)):
            if i == 0:
                continue
            _segments.append([last, i])
            last = i

    postprocess()
