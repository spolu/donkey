import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

import donkey

INPUTS_SIZE = 3 + donkey.ANGLES_WINDOW

class Policy(nn.Module):
    def __init__(self, config):
        super(Policy, self).__init__()
        self.hidden_size = config.get('hidden_size')
        self.config = config

        self.fc1_a = nn.Linear(INPUTS_SIZE, self.hidden_size)
        self.fc2_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3_a = nn.Linear(self.hidden_size, 2 * donkey.CONTROL_SIZE)

        self.fc1_v = nn.Linear(INPUTS_SIZE, self.hidden_size)
        self.fc2_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3_v = nn.Linear(self.hidden_size, 1)

        self.train()

        nn.init.xavier_normal(self.fc1_a.weight.data, nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal(self.fc2_a.weight.data, nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal(self.fc3_a.weight.data, nn.init.calculate_gain('tanh'))
        self.fc1_a.bias.data.fill_(0)
        self.fc2_a.bias.data.fill_(0)
        self.fc3_a.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc1_v.weight.data, nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal(self.fc2_v.weight.data, nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal(self.fc3_v.weight.data, nn.init.calculate_gain('linear'))
        self.fc1_v.bias.data.fill_(0)
        self.fc2_v.bias.data.fill_(0)
        self.fc3_v.bias.data.fill_(0)

    def inputs_size(self):
        return INPUTS_SIZE

    def preprocess(self, observation):
        track_angles = [o.track_angles for o in observation]
        track_position = [[o.track_position] for o in observation]
        track_linear_speed = [[o.track_linear_speed] for o in observation]
        time = [[o.time] for o in observation]

        observation = np.concatenate(
            (
                np.stack(track_angles),
                np.stack(track_position),
                np.stack(track_linear_speed),
                np.stack(time),
            ),
            axis=-1,
        )
        observation = torch.from_numpy(observation).float()

        return observation

    def action(self, inputs, hiddens, masks, deterministic=False):
        value, x, hiddens = self(inputs, hiddens, masks)

        slices = torch.split(x, donkey.CONTROL_SIZE, 1)
        action_mean = slices[0]
        action_logstd = slices[1]
        action_std = action_logstd.exp()

        m = Normal(action_mean, action_std)

        if deterministic is False:
            actions = m.sample()
        else:
            actions = action_mean

        # log_probs (sum on actions) -> batch x 1
        log_probs = m.log_prob(actions).sum(-1, keepdim=True)

        # entropy (sum on actions / mean on batch) -> 1x1
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        entropy = entropy.sum(-1, keepdim=True)

        return value, actions, hiddens, log_probs, entropy

    def evaluate(self, inputs, hiddens, masks, actions):
        value, x, hiddens = self(inputs, hiddens, masks)

        slices = torch.split(x, donkey.CONTROL_SIZE, 1)
        action_mean = slices[0]
        action_logstd = slices[1]
        action_std = action_logstd.exp()

        m = Normal(action_mean, action_std)

        # log_probs (sum on actions) -> batch x 1
        log_probs = m.log_prob(actions).sum(-1, keepdim=True)

        # entropy (sum on actions / mean on batch) -> 1x1
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        entropy = entropy.sum(-1, keepdim=True)

        return value, hiddens, log_probs, entropy

    def forward(self, inputs, hiddens, masks):
        a = F.tanh(self.fc1_a(inputs))
        a = F.tanh(self.fc2_a(a))
        a = F.tanh(self.fc3_a(a))

        v = F.tanh(self.fc1_v(inputs))
        v = F.tanh(self.fc2_v(v))
        v = self.fc3_v(v)

        return v, a, hiddens
