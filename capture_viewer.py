import sys
import io
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
from track import Track

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

@_app.route('/track/<track>/capture/<capture>/camera/<int:index>.jpeg', methods=['GET'])
def camera(track, capture, index):
    capture = fetch_capture(capture)

    if capture.size() == 0:
        abort(400)
    if index > capture.size()-1:
        abort(400)
    if 'camera' not in capture.get_item(index):
        abort(400)

    ib = capture.get_item(index)['camera']

    return send_file(
        io.BytesIO(ib),
        attachment_filename='%d.jpeg' % index,
        mimetype='image/jpeg',
    )

@_app.route('/track/<track>/capture/<capture>/annotate/<int:index>/landmark/<int:landmark>', methods=['GET'])
def annotate(track, capture, index, landmark):
    capture = fetch_capture(capture)
    t = Track(track)

    if capture.size() == 0:
        abort(400)
    if index > capture.size()-1:
        abort(400)
    if 'camera' not in capture.get_item(index):
        abort(400)

    capture.update_item(index, {
        'annotated_track_progress': float(landmark)/TRACK_POINTS,
        'annotated_track_position': 0.0,
    }, save=True)

    return jsonify({})

@_app.route('/track/<track>/capture/<capture>/annotated', methods=['GET'])
def annotated(track, capture):
    capture = fetch_capture(capture)

    if capture.size() == 0:
        abort(400)

    t = Track(track)

    annotated = []
    for i in range(capture.size()):
        if ('annotated_track_progress' in capture.get_item(i) and
                'annotated_track_position' in capture.get_item(i)):
            annotated.append(
                t.invert(
                    capture.get_item(i)['integrated_track_progress'],
                    capture.get_item(i)['integrated_track_position'],
                ).tolist()
            )

    return jsonify(annotated)

@_app.route('/track/<track>/capture/<capture>/integrated', methods=['GET'])
def integrated(track, capture):
    capture = fetch_capture(capture)

    if capture.size() == 0:
        abort(400)

    if 'integrated_track_progress' not in capture.get_item(0):
        abort(400)

    t = Track(track)

    c = []
    for i in range(capture.size()):
        if 'integrated_track_progress' in capture.get_item(i):
            c.append(t.invert(
                capture.get_item(i)['integrated_track_progress'],
                capture.get_item(i)['integrated_track_position'],
            ).tolist())

    return jsonify(c)

@_app.route('/track/<track>/capture/<capture>/corrected', methods=['GET'])
def corrected(track, capture):
    capture = fetch_capture(capture)

    if capture.size() == 0:
        abort(400)

    if 'corrected_track_progress' not in capture.get_item(0):
        abort(400)

    t = Track(track)

    return jsonify([
        t.invert(
            capture.get_item(i)['corrected_track_progress'],
            capture.get_item(i)['corrected_track_position'],
        ).tolist() for i in range(capture.size())
    ])


@_app.route('/track/<track>/capture/<capture>/reference', methods=['GET'])
def reference(track, capture):
    capture = fetch_capture(urllib.parse.unquote(capture))

    if capture.size() == 0:
        abort(400)

    if 'reference_track_progress' not in capture.get_item(0):
        abort(400)

    t = Track(track)

    return jsonify([
        t.invert(
            capture.get_item(i)['reference_track_progress'],
            capture.get_item(i)['reference_track_position'],
        ).tolist() for i in range(capture.size())
    ])

@_app.route('/track/<track>/path', methods=['GET'])
def track(track):
    t = Track(track)

    return jsonify({
      'center': [t.invert(float(p)/TRACK_POINTS, 0).tolist() for p in range(TRACK_POINTS)],
      'right': [t.invert(float(p)/TRACK_POINTS, 1.0).tolist() for p in range(TRACK_POINTS)],
      'left': [t.invert(float(p)/TRACK_POINTS, -1.0).tolist() for p in range(TRACK_POINTS)],
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--capture_set_dir', type=str, help="path to captured data")

    args = parser.parse_args()
    if args.capture_set_dir is not None:
        _capture_set_dir = args.capture_set_dir

    t = threading.Thread(target = run_server)
    t.start()
    t.join()
