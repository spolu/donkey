import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.distributions import Normal

class Runner:
    def __init__(self, config, policy, load_dir):
        self.load_dir = load_dir
        self.hidden_size = config.get('hidden_size')

        self.policy = policy

        self.policy.load_state_dict(
            torch.load(self.load_dir + "/policy.pt", map_location='cpu'),
        )
        self.policy.eval()

    def run(self, img_arr):
        img_arr = img_arr.transpose(2, 0, 1) #transposing the images channels 
        img_arr = img_arr / 127.5 - 1
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        observation = torch.from_numpy(img_arr).float()

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

        return actions[0][0], actions[0][1]

    def shutdown(self):
        pass
