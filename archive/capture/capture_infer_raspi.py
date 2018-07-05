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

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from utils import Config, str2bool, Meter
from capture import Capture
from track import Track
from capture.models import ConvNet

_capture = None
_track = None
GAMMA = 1/5

def test_model(cfg):
    torch.manual_seed(cfg.get('seed'))
    random.seed(cfg.get('seed'))

    if cfg.get('cuda'):
        torch.cuda.manual_seed(cfg.get('seed'))

    device = torch.device('cuda:0' if cfg.get('cuda') else 'cpu')
    model = ConvNet(cfg).to(device)

    if not args.load_dir:
        raise Exception("Required argument: --load_dir")

    if cfg.get('cuda'):
        model.load_state_dict(
            torch.load(args.load_dir + "/model.pt"),
        )
    else:
        model.load_state_dict(
            torch.load(args.load_dir + "/model.pt", map_location='cpu'),
        )

    model.eval()

    last_position = None
    last_index = _capture.size()
    running_position = None

    for i in range(len(_capture.ready)):
        camera = _capture.get_item(_capture.ready[i])['input']
        out = model(camera.unsqueeze(0))

        track_coordinates = out[0].data.numpy().tolist()
        position = _track.invert(np.array(track_coordinates))

        if running_position is None:
            running_position = position

        if last_position is not None:
            next_index = _capture.ready[i]
            for j in range(last_index, next_index):
                p = (j-last_index)/(next_index-last_index)*position + \
                    (next_index-j)/(next_index-last_index)*last_position
                running_position = (1-GAMMA) * running_position + GAMMA * p
                track_coordinates = _track.coordinates(running_position)
                _capture.update_item(j, {
                    'inferred_track_coordinates': track_coordinates.tolist(),
                }, save=False)

        last_position = position
        last_index = _capture.ready[i]


        # print("INFERRED {} {}".format(
        #     _capture.ready[i], track_coordinates,
        # ))
        # _capture.update_item(_capture.ready[i], {
        #     'inferred_track_coordinates': track_coordinates,
        # }, save=False)

    _capture.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--capture_dir', type=str, help="path to saved captured data")
    parser.add_argument('--track', type=str, help="track name")

    parser.add_argument('--load_dir', type=str, help="path to saved models directory")
    parser.add_argument('--cuda', type=str2bool, help="config override")

    args = parser.parse_args()

    cfg = Config('configs/capture_trainer_stack.json')

    if args.cuda != None:
        cfg.override('cuda', args.cuda)

    assert args.capture_dir is not None
    _capture = Capture(args.capture_dir, load=True)

    assert args.track is not None
    _track = Track(args.track)

    test_model(cfg)
