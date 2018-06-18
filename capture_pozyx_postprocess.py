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

GAMMA = 1/10

def postprocess():
    # Integrate the path and update the _capture.
    print("Starting pozyx integration...")

    last_position = None
    last_index = _capture.size()
    running_position = None

    for i in reversed(range(_capture.size())):
        if 'raspi_pozyx_position' in _capture.get_item(i):
            next_position = np.array(_capture.get_item(i)['raspi_pozyx_position'])

            print("Processing position {}".format(i))

            if last_position is not None:
                for j in reversed(range(i, last_index)):
                    p = (j-i)/(last_index-i)*last_position + \
                        (last_index-j)/(last_index-i)*next_position
                    track_coordinates = _track.coordinates(p)
                    _capture.update_item(j, {
                        'integrated_track_coordinates': track_coordinates.tolist(),
                    }, save=False)

                    if running_position is None:
                        running_position = p
                    else:
                        running_position = (1-GAMMA) * running_position + GAMMA * p
                    track_coordinates = _track.coordinates(running_position)
                    _capture.update_item(j, {
                        'corrected_track_coordinates': track_coordinates.tolist(),
                    }, save=False)

            last_position = next_position
            last_index = i



    # for i in reversed(range(_capture.size())):
    #     if last_position is not None:
    #     else:
    #         if 'raspi_pozyx_position' in _capture.get_item(i):
    #             position = np.array(_capture.get_item(i)['raspi_pozyx_position'])
    #             track_advance = _track.advance(position)
    #             track_position = _track.position(position)
    #             track_progress = _track.progress(position)

    #             last_position = track.position

    #         if 'integrated_track_progress' in _capture.get_item(i):
    #             last_position = _capture.get_item(i)['integrated_track_progress']
    #             last_progress = _capture.get_item(i)['integrated_track_progress']


    #     if last_position is not None:
    #         _capture.update_item(i, {
    #             'corrected_track_progress': _track.progress(last_position).tolist(),
    #             'corrected_track_position': _track.position(last_position),
    #         }, save=False)

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
