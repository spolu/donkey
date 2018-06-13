import time
import cv2

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from capture.models import ConvNet

class Localizer:
    def __init__(self, cfg, policy, load_dir):
        self.device = torch.device('cpu')
        self.model = ConvNet(cfg, 3, 3).to(self.device)
        self.stack = None

        if not load_dir:
            raise Exception("Required argument: --load_dir")

        self.model.load_state_dict(
            torch.load(load_dir + "/model.pt", map_location='cpu'),
        )

    def run(self, img_array = None):
        b,g,r = cv2.split(img_array)       # get b,g,r as cv2 uses BGR and not RGB for colors
        rgb_img = cv2.merge([r,g,b])
        camera = cv2.imencode(".jpg", rgb_img)[1].tostring()

        tensor = torch.tensor(cv2.imdecode(
            np.fromstring(camera, np.uint8),
            cv2.IMREAD_COLOR,
        ), dtype=torch.float).to(self.device)
        tensor = tensor / 127.5 - 1
        tensor = tensor.transpose(0, 2)

        output = self.model(tensor.unsqueeze(0))

        track_progress = output[0][0].item()
        track_position = output[0][1].item()
        track_angle = output[0][2].item()

        print(">> LOCALIZER {:.2f} {:.2f} {:.2f}".format(
            track_progress, track_position, track_angle,
        ))

        return track_progress, track_position, track_angle
