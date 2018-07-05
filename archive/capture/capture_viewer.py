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

    camera = cv2.imdecode(
        np.fromstring(ib, np.uint8),
        cv2.IMREAD_GRAYSCALE,
    ).astype(np.float)[50:]
    edges = cv2.Canny(
        camera.astype(np.uint8), 50, 150, apertureSize = 3,
    )
    _, encoded = cv2.imencode('.jpeg', edges)

    print("ENCODED!")
    # import pdb; pdb.set_trace()

    return send_file(
        io.BytesIO(encoded.tobytes()),
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

    track_coordinates = t.coordinates_from_progress(
        float(landmark)/TRACK_POINTS,
        0.0,
    )

    capture.update_item(index, {
        'annotated_track_coordinates': track_coordinates,
    }, save=True)

    return jsonify({})

@_app.route('/track/<track>/capture/<capture>/annotated', methods=['GET'])
def annotated(track, capture):
    capture = fetch_capture(capture)

    if capture.size() == 0:
        abort(400)

    t = Track(track)

    annotated = []
    indices = []
    for i in range(capture.size()):
        if 'annotated_track_coordinates' in capture.get_item(i):
            indices.append(i)
            annotated.append(
                t.invert(
                    capture.get_item(i)['integrated_track_coordinates'],
                ).tolist()
            )
            indices.append(i)
            annotated.append(
                t.invert(
                    capture.get_item(i)['corrected_track_coordinates'],
                ).tolist()
            )

    return jsonify({
        'annotated': annotated,
        'indices': indices,
    })

@_app.route('/track/<track>/capture/<capture>/integrated', methods=['GET'])
def integrated(track, capture):
    capture = fetch_capture(capture)

    if capture.size() == 0:
        abort(400)

    t = Track(track)

    integrated = []
    indices = []

    for i in range(capture.size()):
        if 'integrated_track_coordinates' in capture.get_item(i):
            indices.append(i)
            integrated.append(t.invert(
                capture.get_item(i)['integrated_track_coordinates'],
            ).tolist())

    return jsonify({
        'integrated': integrated,
        'indices': indices,
    })

@_app.route('/track/<track>/capture/<capture>/inferred', methods=['GET'])
def inferred(track, capture):
    capture = fetch_capture(capture)

    if capture.size() == 0:
        abort(400)

    t = Track(track)

    inferred = []
    indices = []

    for i in range(capture.size()):
        if 'inferred_track_coordinates' in capture.get_item(i):
            indices.append(i)
            inferred.append(t.invert(
                capture.get_item(i)['inferred_track_coordinates'],
            ).tolist())

    return jsonify({
        'inferred': inferred,
        'indices': indices,
    })

@_app.route('/track/<track>/capture/<capture>/corrected', methods=['GET'])
def corrected(track, capture):
    capture = fetch_capture(capture)

    if capture.size() == 0:
        abort(400)

    t = Track(track)

    corrected = []
    indices = []

    for i in range(capture.size()):
        if 'corrected_track_coordinates' in capture.get_item(i):
            indices.append(i)
            corrected.append(t.invert(
                capture.get_item(i)['corrected_track_coordinates'],
            ).tolist())

    return jsonify({
        'corrected': corrected,
        'indices': indices,
    })

@_app.route('/track/<track>/path', methods=['GET'])
def track(track):
    t = Track(track)

    return jsonify({
      'center': [t.invert(t.coordinates_from_progress(float(p)/TRACK_POINTS, 0)).tolist() for p in range(TRACK_POINTS)],
      'right': [t.invert(t.coordinates_from_progress(float(p)/TRACK_POINTS, 1.0)).tolist() for p in range(TRACK_POINTS)],
      'left': [t.invert(t.coordinates_from_progress(float(p)/TRACK_POINTS, -1.0)).tolist() for p in range(TRACK_POINTS)],
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
