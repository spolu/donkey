import sys
import os
import argparse
import eventlet
import eventlet.wsgi
import math

import capture

import numpy as np
from flask import Flask
from flask import jsonify
from eventlet.green import threading

from capture import Capture
from simulation import Track

# import pdb; pdb.set_trace()

TRACK_POINTS = 400
CAPTURE_ROOT_PATH = '/tmp'

_app = Flask(__name__)
_cache = {}

def fetch_capture(capture):
    if capture not in _cache:
        _cache[capture] = Capture(os.path.join(CAPTURE_ROOT_PATH, capture))
    return _cache[capture]

def run_server():
    global _app

    print("Starting shared server: port=9091")
    address = ('0.0.0.0', 9091)
    try:
        eventlet.wsgi.server(eventlet.listen(address), _app)
    except KeyboardInterrupt:
        print("Stopping shared server")

# @_app.route('/capture/<track>', methods=['GET'])
# def capture(track):
#     t = Track(track)
#
#     time = [_capture.get_item(i)['time'] for i in range(_capture.__len__())]
#     angular_velocity = [_capture.get_item(i)['angular_velocity'] for i in range(_capture.__len__())]
#     acceleration = [_capture.get_item(i)['acceleration'] for i in range(_capture.__len__())]
#
#     angle = [math.pi]
#     for i in range(1, len(time)):
#         angle.append(angle[i-1] + angular_velocity[i][1] * (time[i] - time[i-1]))
#
#     velocity = [np.array([0.0, 0.0, 0.0])]
#     for i in range(1, len(time)):
#         # print("ACCELERATIOn {}".format(acceleration[i]))
#         print("TIME {}".format((time[i] - time[i-1])))
#         velocity.append(velocity[i-1] + np.array(acceleration[i]) * (time[i] - time[i-1]))
#
#     positions = [np.array([0,0,0])]
#     # for i in range(1, len(time)):
#     #     positions.append(
#     #         positions[i-1] + [
#     #             velocity[i-1][0] * (time[i] - time[i-1]) * 0.8,
#     #             velocity[i-1][1] * (time[i] - time[i-1]) * 0.8,
#     #             velocity[i-1][2] * (time[i] - time[i-1]) * 0.8,
#     #         ],
#     #     )
#     for i in range(1, len(time)):
#         velocity[i-1][1] = 0.0
#         speed = np.linalg.norm(velocity[i-1])
#         # print("SPEED {}".format(speed))
#         positions.append(
#             positions[i-1] + [
#                 -np.sin(angle[i-1]) * speed * (time[i] - time[i-1]),
#                 0,
#                 -np.cos(angle[i-1]) * speed * (time[i] - time[i-1]),
#             ],
#         )
#     for i in range(len(positions)):
#         positions[i] = positions[i].tolist()
#
#     return jsonify(positions)

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
                t.invert_position(
                    capture.get_item(i)['annotated_progress'],
                    capture.get_item(i)['annotated_track_position'],
                ).tolist()
            )

    return jsonify(annotated)

@_app.route('/track/<track>/capture/<capture>/computed', methods=['GET'])
def computed(track, capture):
    capture = fetch_capture(capture)

    if capture.__len__() == 0:
        abort(400)

    if 'computed_progress' not in capture.get_item(0):
        abort(400)

    t = Track(track)

    return jsonify([
        t.invert_position(
            capture.get_item(i)['computed_progress'],
            capture.get_item(i)['computed_track_position'],
        ).tolist() for i in range(capture.__len__())
    ])

@_app.route('/track/<track>/capture/<capture>/reference', methods=['GET'])
def reference(track, capture):
    capture = fetch_capture(capture)

    if capture.__len__() == 0:
        abort(400)

    if 'reference_progress' not in capture.get_item(0):
        abort(400)

    t = Track(track)

    return jsonify([
        t.invert_position(
            capture.get_item(i)['reference_progress'],
            capture.get_item(i)['reference_track_position'],
        ).tolist() for i in range(capture.__len__())
    ])

@_app.route('/track/<track>/path', methods=['GET'])
def track(track):
    t = Track(track)

    return jsonify({
      'center': [t.invert_position(float(p)/TRACK_POINTS, 0).tolist() for p in range(TRACK_POINTS)],
      'left': [t.invert_position(float(p)/TRACK_POINTS, 5.0).tolist() for p in range(TRACK_POINTS)],
      'right': [t.invert_position(float(p)/TRACK_POINTS, -5.0).tolist() for p in range(TRACK_POINTS)],
    })

if __name__ == "__main__":
    t = threading.Thread(target = run_server)
    t.start()
    t.join()
