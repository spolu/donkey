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

def clear_annotation():
    for i in range(_capture.size()):
        _capture.update_item(i, {
            'annotated_track_progress': None,
            'annotated_track_position': None,
        }, save=False)

    _capture.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--capture_dir', type=str, help="path to saved captured data")

    args = parser.parse_args()

    assert args.capture_dir is not None
    _capture = Capture(args.capture_dir, load=True)

    clear_annotation()
