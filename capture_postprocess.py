import sys
import os
import argparse
import eventlet
import eventlet.wsgi

import capture

from flask import Flask
from flask import jsonify
from eventlet.green import threading

from capture import Capture
from simulation import Track

TRACK_POINTS = 400

_app = Flask(__name__)
_capture = None
_track = None

def run_server():
    global _app

    print("Starting shared server: port=9091")
    address = ('0.0.0.0', 9091)
    try:
        eventlet.wsgi.server(eventlet.listen(address), _app)
    except KeyboardInterrupt:
        print("Stopping shared server")

@_app.route('/capture/<speed>', methods=['GET'])
def capture(speed):
    pass

@_app.route('/track/<track>', methods=['GET'])
def track(track):
    t = Track(track)
    return jsonify({
      'center': [t.invert_position(float(p)/TRACK_POINTS, 0).tolist() for p in range(TRACK_POINTS)],
      'left': [t.invert_position(float(p)/TRACK_POINTS, 5.0).tolist() for p in range(TRACK_POINTS)],
      'right': [t.invert_position(float(p)/TRACK_POINTS, -5.0).tolist() for p in range(TRACK_POINTS)],
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--capture_dir', type=str, help="path to saved captured data")

    args = parser.parse_args()

    assert args.capture_dir is not None
    _capture = Capture(args.capture_dir)

    t = threading.Thread(target = run_server)
    t.start()
    t.join()
