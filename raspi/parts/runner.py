import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.distributions import Normal

CAMERA_STACK_SIZE = 3

class Runner:
    def __init__(self, config, policy, load_dir):
        self.load_dir = load_dir
        self.hidden_size = config.get('hidden_size')

        self.policy = policy

        self.policy.load_state_dict(
            torch.load(self.load_dir + "/policy.pt", map_location='cpu'),
        )
        self.policy.eval()
        self.camera_stack = None

    def run(self, img_stack):
        # Transpose the images channels.
        img_stack = img_stack.transpose(2, 0, 1)
        img_stack = img_stack.reshape((1,) + img_stack.shape)
        img_stack = img_stack / 127.5 - 1

        observation = torch.from_numpy(img_stack).float()

        value, x, auxiliary, hiddens = self.policy(
            autograd.Variable(
                observation, requires_grad=False,
            ),
            autograd.Variable(
                torch.zeros(1, self.hidden_size), requires_grad=False,
            ),
            autograd.Variable(
                torch.ones(1), requires_grad=False,
            ),
        )

        slices = torch.split(x, 2, 1)
        actions = slices[0].data.numpy()

        print("ACTIONS {:.2f} / {:.2f}".format(actions[0][0], actions[0][1]))

        return actions[0][0], min(0.60, actions[0][1])

    def shutdown(self):
        pass
