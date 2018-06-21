import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class WMRNN(nn.Module):
    def __init__(self, config):
        super(WMRNN, self).__init__()

        self.gaussians_count = config.get('world_model_rnn_gaussians_count')
        self.z_size = config.get('world_model_z_size')

        self.layers_count = config.get('world_model_rnn_layers_count')
        self.hidden_size = config.get('world_model_hidden_size')

        # Encoding.
        self.fc1 = nn.Linear(self.z_size + 1, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.layers_count)

        # Output.
        self.fc_pi = nn.Linear(self.hidden_size, self.gaussians_count * self.z_size)
        self.fc_sigma = nn.Linear(self.hidden_size, self.gaussians_count * self.z_size)
        self.fc_mu = nn.Linear(self.hidden_size, self.gaussians_count * self.z_size)

    def forward(self, x, h):
        self.lstm.flatten_parameters()

        x = F.relu(self.fc1(x))
        z, h = self.lstm(x, h)

        pi = F.softmax(
            self.fc_pi(z).view(
                -1, x.size()[1], self.gaussians_count, self.z_size,
            ),
            dim=2,
        )
        sigma = torch.exp(
            self.fc_sigma(z).view(
                -1,  x.size()[1], self.n_gaussians, self.z_dim,
            ),
        )
        mu = self.fc_mu(z).view(
            -1, x.size()[1], self.n_gaussians, self.z_dim,
        )

        return pi, sigma, mu


