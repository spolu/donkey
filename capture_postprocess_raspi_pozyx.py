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

def postprocess_raspi_pozyx():
    # Integrate the path and update the _capture.
    print("Starting pozyx heuristic...")

    self.position = np.array([0,0,0])

    for i in range(_capture.size()):
        if 'raspi_pozyx_position' in _capture.get_item(i):
            last = np.array(_capture.get_item(i)['raspi_pozyx_position'])
            distance = np.linalg.norm(last - self.position)




    angles, positions = integrate(
        0, _capture.size(),
        math.pi,
        np.array([0,0,0]),
        _speed,
    )
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

    postprocess_pozyx_raspi()
