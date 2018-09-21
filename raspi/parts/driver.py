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
        self.start_time = time.time()
        self.throttle = self.driver_fixed_throttle

    def run(self, camera = None, flow_speed = None):
        camera = self.input_filter.apply(camera) / 127.5 - 1

        camera = torch.from_numpy(camera).float().unsqueeze(0).to(self.device)

        _, action, hiddens, _, _ = self.policy.action(
            camera.unsqueeze(0).detach(),
            self.hiddens.detach(),
            self.masks.detach(),
            deterministic=True,
        )

        steering = action[0][0].item()
        throttle = self.throttle

        # Step increase of throttle command.
        # if time.time() - self.start_time > 30.0:
        #     self.start_time = time.time()
        #     if self.throttle < 0.65:
        #         self.throttle += 0.01

        SPEED = 22

        if flow_speed < SPEED-5:
            self.throttle += 0.001
        elif flow_speed < SPEED:
            self.throttle += 0.0005

        if flow_speed > SPEED:
            self.throttle -= 0.0005
        elif flow_speed > SPEED+5:
            self.throttle -= 0.001

        print(">>> COMMANDS: {:.2f} {:.2f}".format(
            steering, throttle
        ))

        return steering, throttle

