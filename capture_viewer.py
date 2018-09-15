import sys
import io
import cv2
import os
import argparse
import eventlet
import eventlet.wsgi
import math
import urllib

import capture

import numpy as np
from flask import Flask
from flask import jsonify
from flask import abort
from flask import send_file
from eventlet.green import threading

from capture import Capture

# import pdb; pdb.set_trace()

TRACK_POINTS = 400

_app = Flask(__name__)
_capture_set_dir = '/tmp'
_cache = {}

def fetch_capture(capture):
    if capture not in _cache:
        _cache[capture] = Capture(os.path.join(_capture_set_dir, capture))
    return _cache[capture]

def run_server():
    global _app

    print("Starting shared server: port=9092")
    address = ('0.0.0.0', 9092)
    try:
        eventlet.wsgi.server(eventlet.listen(address), _app)
    except KeyboardInterrupt:
        print("Stopping shared server")

@_app.route('/capture/<capture>/camera/<int:index>.jpeg', methods=['GET'])
def camera( capture, index):
    capture = fetch_capture(capture)

    if capture.size() == 0:
        abort(400)
    if index > capture.size()-1:
        abort(400)
    if 'camera' not in capture.get_item(index):
        abort(400)

    ib = capture.get_item(index)['camera']

    camera = cv2.imdecode(
        np.fromstring(ib, np.uint8),
        cv2.IMREAD_GRAYSCALE,
    ).astype(np.float)[50:]
    edges = cv2.Canny(
        camera.astype(np.uint8), 200, 250, apertureSize = 3,
    )
    _, encoded = cv2.imencode('.jpeg', edges)

    # import pdb; pdb.set_trace()

    return send_file(
        io.BytesIO(encoded.tobytes()),
        attachment_filename='%d.jpeg' % index,
        mimetype='image/jpeg',
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--capture_set_dir', type=str, help="path to captured data")

    args = parser.parse_args()
    if args.capture_set_dir is not None:
        _capture_set_dir = args.capture_set_dir

    t = threading.Thread(target = run_server)
    t.start()
    t.join()
