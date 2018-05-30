import sys
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
from eventlet.green import threading

from capture import Capture
from track import Track

# import pdb; pdb.set_trace()

TRACK_POINTS = 400
CAPTURE_ROOT_PATH = '/tmp'
OFF_TRACK_DISTANCE = 6.0

_app = Flask(__name__)
_cache = {}

def fetch_capture(capture):
    if capture not in _cache:
        _cache[capture] = Capture(os.path.join(CAPTURE_ROOT_PATH, capture))
    return _cache[capture]

def run_server():
    global _app

    print("Starting shared server: port=9092")
    address = ('0.0.0.0', 9092)
    try:
        eventlet.wsgi.server(eventlet.listen(address), _app)
    except KeyboardInterrupt:
        print("Stopping shared server")

@_app.route('/track/<track>/capture/<capture>/annotated', methods=['GET'])
def annotated(track, capture):
    capture = fetch_capture(capture)

    if capture.__len__() == 0:
        abort(400)

    t = Track(track)
    annotated = []

    for i in range(capture.__len__()):
        if 'annotated_progress' in capture.get_item(i) and 'annotated_track_position' in capture.get_item(i):
            annotated.append(
                t.invert(
                    capture.get_item(i)['annotated_progress'],
                    capture.get_item(i)['annotated_track_position'] * OFF_TRACK_DISTANCE,
                ).tolist()
            )

    return jsonify(annotated)

@_app.route('/track/<track>/capture/<capture>/integrated', methods=['GET'])
def integrated(track, capture):
    capture = fetch_capture(capture)

    if capture.__len__() == 0:
        abort(400)

    if 'integrated_progress' not in capture.get_item(0):
        abort(400)

    t = Track(track)

    c = []
    for i in range(capture.__len__()):
        if 'integrated_progress' in capture.get_item(i):
            c.append(t.invert(
                capture.get_item(i)['integrated_progress'],
                capture.get_item(i)['integrated_track_position'] * OFF_TRACK_DISTANCE,
            ).tolist())

    return jsonify(c)

@_app.route('/track/<track>/capture/<capture>/corrected', methods=['GET'])
def corrected(track, capture):
    capture = fetch_capture(capture)

    if capture.__len__() == 0:
        abort(400)

    if 'corrected_progress' not in capture.get_item(0):
        abort(400)

    t = Track(track)

    return jsonify([
        t.invert(
            capture.get_item(i)['corrected_progress'],
            capture.get_item(i)['corrected_track_position'] * OFF_TRACK_DISTANCE,
        ).tolist() for i in range(capture.__len__())
    ])


@_app.route('/track/<track>/capture/<capture>/reference', methods=['GET'])
def reference(track, capture):
    capture = fetch_capture(urllib.parse.unquote(capture))

    if capture.__len__() == 0:
        abort(400)

    if 'reference_progress' not in capture.get_item(0):
        abort(400)

    t = Track(track)

    return jsonify([
        t.invert(
            capture.get_item(i)['reference_progress'],
            capture.get_item(i)['reference_track_position'] * OFF_TRACK_DISTANCE,
        ).tolist() for i in range(capture.__len__())
    ])

@_app.route('/track/<track>/path', methods=['GET'])
def track(track):
    t = Track(track)

    return jsonify({
      'center': [t.invert(float(p)/TRACK_POINTS, 0).tolist() for p in range(TRACK_POINTS)],
      'left': [t.invert(float(p)/TRACK_POINTS, 1.0).tolist() for p in range(TRACK_POINTS)],
      'right': [t.invert(float(p)/TRACK_POINTS, -1.0).tolist() for p in range(TRACK_POINTS)],
    })

if __name__ == "__main__":
    t = threading.Thread(target = run_server)
    t.start()
    t.join()
