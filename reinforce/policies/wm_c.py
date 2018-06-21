import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class WMC(nn.Module):
    def __init__(self, config):
        super(WMC, self).__init__()

        self.z_size = config.get('world_model_z_size')
        self.hidden_size = config.get('world_model_hidden_size')

        self.fc1_v = nn.Linear(self.hidden_size + self.z_size, self.hidden_size)
        self.fc2_v = nn.Linear(self.hidden_size, 1)

        self.fc1_a = nn.Linear(self.hidden_size, self.hidden_size)
        if self.action_type == 'discrete':
            self.fc2_a = nn.Linear(self.hidden_size, reinforce.DISCRETE_CONTROL_SIZE)
        else:
            self.fc2_a = nn.Linear(self.hidden_size, 2 * reinforce.CONTINUOUS_CONTROL_SIZE)

        nn.init.xavier_normal_(self.fc1_a.weight.data, nn.init.calculate_gain('relu'))
        if self.action_type == 'discrete':
            nn.init.xavier_normal_(self.fc2_a.weight.data, nn.init.calculate_gain('linear'))
        else:
            nn.init.xavier_normal_(self.fc2_a.weight.data, nn.init.calculate_gain('relu'))
        self.fc1_a.bias.data.fill_(0)
        self.fc2_a.bias.data.fill_(0)

        nn.init.xavier_normal_(self.fc1_v.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc2_v.weight.data, nn.init.calculate_gain('linear'))
        self.fc1_v.bias.data.fill_(0)
        self.fc2_v.bias.data.fill_(0)

    def forward(self, h, z, masks):
        # Value head.
        v = F.relu(self.fc1_v(torch.cat((h, z), 0)))
        v = self.fc2_v(v)

        # Action head.
        a = F.relu(self.fc1_a(toch.cat((h, z), 0)))
        if self.action_type == 'discrete':
            a = self.fc2_a(a)
        else:
            a = F.tanh(self.fc2_a(a))

