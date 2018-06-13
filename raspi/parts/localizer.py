import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from capture.models import ResNet

class Localizer:
    def __init__(self, cfg, policy, load_dir):
        self.device = torch.device('cpu')
        self.stack_size = cfg.get('stack_size')
        self.model = ResNet(cfg, 3 * self.stack_size, 1, 3).to(self.device)

        if not load_dir:
            raise Exception("Required argument: --load_dir")

        self.model.load_state_dict(
            torch.load(load_dir + "/model.pt", map_location='cpu'),
        )

    def run(self, img_array = None):
        width, height, _ = img_array.shape

        if self.stack is None:
            self.stack = torch.zeros(
                self.stack_size * 3, width, height
            ).to(self.device)

        for ch in range(self.stack_size - 1):
            self.stack[ch] = self.stack[ch+1]
        self.stack[self.stack_size-1] = torch.tensor(img_array).to(self.device)

        output = self.model(
            self.stack.unsqueeze(0),
            torch.zeros(self.stack_size, 1).to(self.device),
        )

        track_progress = output[0][0].item()
        track_position = output[0][1].item()
        track_angle = output[0][2].item()

        return track_progress, track_position, track_angle
