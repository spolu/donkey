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

EPSILON = 0.10

def postprocess():
    # Integrate the path and update the _capture.
    print("Starting pozyx heuristic...")

    last_position = np.array([0,0,0])
    last_advance = _track.advance(last_position)

    for i in range(_capture.size()):
        if 'raspi_pozyx_position' in _capture.get_item(i):
            position = np.array(_capture.get_item(i)['raspi_pozyx_position'])
            advance = _track.advance(position)
            distance = np.linalg.norm(position - last_position)

            if distance > EPSILON and advance > last_advance:
                last_position = (position + last_position) / 2.0
            elif advance < last_advance:
                last_position = last_position
            else:
                last_position = position
            last_advance = _track.advance(last_position)

            _capture.update_item(i, {
                'integrated_track_progress': _track.progress(position).tolist(),
                'integrated_track_position': _track.position(position),
            }, save=False)

        _capture.update_item(i, {
            'corrected_track_progress': _track.progress(last_position).tolist(),
            'corrected_track_position': _track.position(last_position),
        }, save=False)

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

    postprocess()
