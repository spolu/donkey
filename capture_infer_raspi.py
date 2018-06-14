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

    for i in range(len(_capture.ready)):
        camera = _capture.get_item(_capture.ready[i])['input']
        progress, position = model(camera.unsqueeze(0))

        track_progress = progress[0][0].item()
        track_position = position[0][0].item()
        # track_angle = output[0][2].item()

        print("INFERRED {} {} {}".format(
            _capture.ready[i], track_progress, track_position,
        ))
        _capture.update_item(_capture.ready[i], {
            'inferred_track_progress': track_progress,
            'inferred_track_position': track_position,
            # 'inferred_track_angle': track_angle,
        }, save=False)

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

    test_model(cfg)
