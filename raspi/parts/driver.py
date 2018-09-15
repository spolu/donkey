import time
import cv2

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from track import Track
from reinforce.policies import PPOPixelsCNN
from reinforce.input_filter import InputFilter

# import pdb; pdb.set_trace()

class Driver:
    def __init__(self, cfg, load_dir = None):
        self.track_name = cfg.get('track_name')
        self.hidden_size = cfg.get('hidden_size')
        self.driver_fixed_throttle = cfg.get('driver_fixed_throttle')
        self.device = torch.device('cpu')

        self.input_filter = InputFilter(cfg)

        if cfg.get('policy') == 'ppo_pixels_cnn':
            self.policy = PPOPixelsCNN(cfg).to(self.device)
        assert self.policy is not None
        self.track = Track(self.track_name)

        self.load_dir = cfg.get('reinforce_load_dir')
        assert self.load_dir is not None

        if cfg.get('device') != 'cpu':
            self.policy.load_state_dict(
                torch.load(self.load_dir + "/policy.pt"),
            )
        else:
            self.policy.load_state_dict(
                torch.load(self.load_dir + "/policy.pt", map_location='cpu'),
            )

        self.policy.eval()

        self.hiddens = torch.zeros(1, self.hidden_size).to(self.device)
        self.masks = torch.ones(1, 1).to(self.device)

    def run(self, img_array = None, flow_dx=None, flow_dy=None):
        b,g,r = cv2.split(img_array)       # get b,g,r as cv2 uses BGR and not RGB for colors
        rgb_img = cv2.merge([r,g,b])

        camera_raw = cv2.imencode(".jpg", rgb_img)[1].tostring()
        camera = cv2.imdecode(
            np.fromstring(camera_raw, np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )

        camera = self.input_filter.apply(camera) / 127.5 - 1
        camera = torch.from_numpy(camera).float().unsqueeze(0).to(self.device)
        _, action, hiddens, _, _ = self.policy.action(
            camera.unsqueeze(0).detach(),
            self.hiddens.detach(),
            self.masks.detach(),
            deterministic=True,
        )

        steering = action[0][0].item()
        # steering = 0.0
        throttle = self.driver_fixed_throttle

        print(">>> COMMANDS: {:.2f} {:.2f}".format(
            steering, throttle
        ))

        return steering, throttle

