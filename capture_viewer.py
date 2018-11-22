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

from utils import Config
from capture import Capture
from raspi.parts.dummy import Dummy

# import pdb; pdb.set_trace()

TRACK_POINTS = 400
_canny_low = 0
_canny_high = 0

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
    if index <= 0:
        abort(400)
    if 'camera' not in capture.get_item(index):
        abort(400)

    feature_params = dict( maxCorners = 100,
                          qualityLevel = 0.3,
                          minDistance = 7,
                          blockSize = 7 )

    lk_params = dict( winSize  = (15,15),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    steering = capture.get_item(index)['raspi_steering']

    ib = capture.get_item(index)['camera']
    ib_prev = capture.get_item(index-1)['camera']
    camera = cv2.imdecode(
        np.fromstring(ib, np.uint8),
        cv2.CV_8UC1
    )[50:]

    camera_prev = cv2.imdecode(
        np.fromstring(ib_prev, np.uint8),
        cv2.CV_8UC1
    )[50:]

    p0 = np.array([
        [[50,10]],[[75,10]],[[100, 10]],
        [[50,20]],[[75,20]],[[100, 20]],
        [[10,40]], [[50,40]],[[75,40]],[[100, 40]],[[140, 40]],
    ], dtype=np.float32)
    p1, st, err = cv2.calcOpticalFlowPyrLK(camera_prev, camera, p0, None, **lk_params)

    good_new = p1[st==1]
    good_old = p0[st==1]

    mask = cv2.cvtColor(np.zeros_like(camera),cv2.COLOR_GRAY2RGB)

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (c,d), (a,b), [0,0,255], 2)

    direction_x, direction_y = 80 + int(25*math.sin(steering*3.1415/6.0)), int(35*(2 - math.cos(steering*3.1415/6.0))),
    mask = cv2.line(mask, (direction_x, direction_y), (80,70), [0,255,0], 2)

    edges = cv2.Canny(
        camera.astype(np.uint8), _canny_low, _canny_high, apertureSize = 3,
    )
    print("canny low: {:.0f}, canny high: {:.0f}".format(_canny_low, _canny_high))

    left = np.sum(edges[15:, :80])
    right = np.sum(edges[15:, 80:])
    p = left / (left + right)

    print("p: {:.2f}".format(p))
    # show p
    mask = cv2.line(mask, (40 + int(80*p), 60), (40 + 40,60), [255,0,0], 2)
    # reference line for p
    mask = cv2.line(mask, (40, 61), (40 + 80,61), [128,128,128], 1)

    backtorgb = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
    backtorgb = cv2.add(backtorgb, mask)

    _, encoded = cv2.imencode('.jpeg', backtorgb)

    # _, encoded = cv2.imencode('.jpeg', camera)

    # import pdb; pdb.set_trace()

    return send_file(
        io.BytesIO(encoded.tobytes()),
        attachment_filename='%d.jpeg' % index,
        mimetype='image/jpeg',
    )

@_app.route('/capture/<capture>/raw/<int:index>.jpeg', methods=['GET'])
def raw( capture, index):

    capture = fetch_capture(capture)

    if capture.size() == 0:
        abort(400)
    if index > capture.size()-1:
        abort(400)
    if index <= 0:
        abort(400)
    if 'camera' not in capture.get_item(index):
        abort(400)

    feature_params = dict( maxCorners = 100,
                          qualityLevel = 0.3,
                          minDistance = 7,
                          blockSize = 7 )

    lk_params = dict( winSize  = (15,15),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    ib = capture.get_item(index)['camera']
    camera = cv2.imdecode(
        np.fromstring(ib, np.uint8),
        cv2.CV_8UC1
    )
    
    _, encoded = cv2.imencode('.jpeg', camera)

    # _, encoded = cv2.imencode('.jpeg', camera)

    # import pdb; pdb.set_trace()

    return send_file(
        io.BytesIO(encoded.tobytes()),
        attachment_filename='%d.jpeg' % index,
        mimetype='image/jpeg',
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--config_path', type=str, help="path to the config file")
    parser.add_argument('--capture_set_dir', type=str, help="path to captured data")
    parser.add_argument('--canny_low', type=int, help="low in canny parameters, all points below are not detected")
    parser.add_argument('--canny_high', type=int, help="high in canny parameters, all points above are detected")

    args = parser.parse_args()

    cfg = Config(args.config_path)
    if args.canny_low != None:
        cfg.override('input_filter_canny_low', args.canny_low)
    if args.canny_high != None:
        cfg.override('input_filter_canny_high', args.canny_high)

    if args.capture_set_dir is not None:
        _capture_set_dir = args.capture_set_dir
    if args.canny_low is not None:
        _canny_low = args.canny_low
    if args.canny_high is not None:
        _canny_high = args.canny_high

    t = threading.Thread(target = run_server)
    t.start()
    t.join()
