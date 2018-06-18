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
_capture_dir = None

def cleanup():
    for i in range(_capture.size()):
        if 'camera' in _capture.get_item(i) and 'raspi_steering' not in _capture.get_item(i):
            print("CLEANUP {}".format(str(_capture.offset + i) + '.jpeg'))
            os.remove(os.path.join(_capture_dir, str(_capture.offset + i) + '.jpeg'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--capture_dir', type=str, help="path to saved captured data")

    args = parser.parse_args()

    assert args.capture_dir is not None
    _capture = Capture(args.capture_dir, load=True)
    _capture_dir = args.capture_dir

    cleanup()
