import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from capture.models import ResNet

class Localizer:
    '''
    Installation:
    sudo apt-get install sense-hat

    '''
    def __init__(self, cfg, policy, load_dir ,poll_delay=0.0166):
        self.on = True
        self.poll_delay = poll_delay
        self.device = torch.device('cpu')
        self.stack_size = cfg.get('stack_size')
        self.model = ResNet(cfg, 3 * self.stack_size, 1, 3).to(self.device)

        if not load_dir:
            raise Exception("Required argument: --load_dir")

        self.model.load_state_dict(
            torch.load(load_dir + "/model.pt", map_location='cpu'),
        )

    def update(self):
        while self.on:
            time.sleep(self.poll_delay)

    def run_threaded(self, img_array = None):
        arr = [img_array for i in range(0, self.stack_size)]
        stack = torch.cat(arr, 0).unsqueeze(0)

        output = self.model(
            stack,
            torch.zeros(self.stack_size, 1).to(self.device),
        )
        track_progress = output[0][0].item()
        track_position = output[0][1].item()
        track_angle = output[0][2].item()
        return track_progress,track_position,track_angle

    def shutdown(self):
        self.on = False

if __name__ == "__main__":
    iter = 0
    while iter < 100:
        iter += 1
