import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.distributions import Normal

import donkey

# import pdb; pdb.set_trace()

class Policy(nn.Module):
    def __init__(self, config):
        super(Policy, self).__init__()
        self.hidden_size = config.get('hidden_size')
        self.recurring_cell = config.get('recurring_cell')
        self.config = config

        self.cv1 = nn.Conv2d(donkey.CAMERA_STACK_SIZE, 32, 8, stride=4)
        self.cv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.cv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc1 = nn.Linear(11264, self.hidden_size)

        if self.recurring_cell == "gru":
            self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)

        # Action network.
        self.ax1_a = nn.Linear(self.hidden_size, donkey.ANGLES_WINDOW)

        self.fc2_a = nn.Linear(self.hidden_size + donkey.ANGLES_WINDOW, self.hidden_size)
        self.fc3_a = nn.Linear(self.hidden_size, 2 * donkey.CONTROL_SIZE)

        # Value network.
        self.fc2_v = nn.Linear(self.hidden_size + donkey.ANGLES_WINDOW, self.hidden_size)
        self.fc3_v = nn.Linear(self.hidden_size, 1)

        self.train()

        # Initialization
        nn.init.xavier_normal(self.cv1.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv2.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv3.weight.data, nn.init.calculate_gain('relu'))
        self.cv1.bias.data.fill_(0)
        self.cv2.bias.data.fill_(0)
        self.cv3.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc1.weight.data, nn.init.calculate_gain('relu'))
        self.fc1.bias.data.fill_(0)

        if self.recurring_cell == "gru":
            nn.init.xavier_normal(self.gru.weight_ih.data)
            nn.init.xavier_normal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        nn.init.xavier_normal(self.ax1_a.weight.data, nn.init.calculate_gain('linear'))
        self.ax1_a.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc2_a.weight.data, nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal(self.fc3_a.weight.data, nn.init.calculate_gain('tanh'))
        self.fc2_a.bias.data.fill_(0)
        self.fc3_a.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc2_v.weight.data, nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal(self.fc3_v.weight.data, nn.init.calculate_gain('linear'))
        self.fc2_v.bias.data.fill_(0)
        self.fc3_v.bias.data.fill_(0)

    def forward(self, inputs, hiddens, masks):
        # A little bit of pytorch magic to extract camera pixels and angles
        # from the packed input.
        pixels_inputs, stack = torch.split(
            inputs, (donkey.CAMERA_STACK_SIZE), dim=1
        )
        angles_inputs = stack.split(1, 2)[0].split(
            (donkey.ANGLES_WINDOW), 3,
        )[0].contiguous().view(-1, donkey.ANGLES_WINDOW)

        x = F.relu(self.cv1(pixels_inputs))
        x = F.relu(self.cv2(x))
        x = F.relu(self.cv3(x))

        x = x.view(-1, 11264)
        x = F.relu(self.fc1(x))

        if self.recurring_cell == "gru":
            if inputs.size(0) == hiddens.size(0):
                x = hiddens = self.gru(x, hiddens * masks)
            else:
                x = x.view(-1, hiddens.size(0), x.size(1))
                masks = masks.view(-1, hiddens.size(0), 1)
                outputs = []
                for i in range(a.size(0)):
                    hx = hiddens = self.gru(x[i], hiddens * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)
            x = F.tanh(x)

        # Action network.
        angles = self.ax1_a(x)

        a = F.tanh(self.fc2_a(torch.cat((x, angles), 1)))
        a = F.tanh(self.fc3_a(a))

        # Value network.
        v = F.tanh(self.fc2_v(torch.cat((x, angles_inputs), 1)))
        v = self.fc3_v(v)

        return v, a, angles, hiddens

    def input_shape(self):
        # We encode the angles of the observation as the first floats of an
        # extra camera layer.
        assert donkey.CAMERA_HEIGHT > donkey.ANGLES_WINDOW

        return (
            donkey.CAMERA_STACK_SIZE + 1,
            donkey.CAMERA_WIDTH,
            donkey.CAMERA_HEIGHT,
        )

    def input(self, observation):
        pack = [np.zeros((
            donkey.CAMERA_STACK_SIZE + 1,
            donkey.CAMERA_WIDTH,
            donkey.CAMERA_HEIGHT,
        ))] * len(observation)
        for o in range(len(observation)):
            for i in range(donkey.CAMERA_STACK_SIZE):
                pack[o][i] = observation[o].camera[i]
            for i in range(donkey.ANGLES_WINDOW):
                pack[o][donkey.CAMERA_STACK_SIZE][0][i] = observation[o].track_angles[i]

        observation = np.concatenate(
            (
                np.stack(pack),
            ),
            axis=-1,
        )
        observation = torch.from_numpy(observation).float()

        return observation

    def auxiliary_shape(self):
        return (donkey.ANGLES_WINDOW,)

    def auxiliary(self, observation):
        track_angles = [o.track_angles for o in observation]

        auxiliary = np.concatenate(
            (
                np.stack(track_angles),
            ),
            axis=-1,
        )
        auxiliary = torch.from_numpy(auxiliary).float()

        return auxiliary

    def action(self, inputs, hiddens, masks, deterministic=False):
        value, x, auxiliary, hiddens  = self(inputs, hiddens, masks)

        slices = torch.split(x, donkey.CONTROL_SIZE, 1)
        action_mean = slices[0]
        action_logstd = slices[1]
        action_std = action_logstd.exp()

        if self.config.get('fixed_action_std'):
            action_std = (
                self.config.get('fixed_action_std') *
                torch.ones(action_mean.size()).float()
            )
            if self.config.get('cuda'):
                action_std = action_std.cuda()
            action_std = autograd.Variable(action_std)
            action_logstd = action_std.log()

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

        return value, actions, auxiliary, hiddens, log_probs, entropy

    def evaluate(self, inputs, hiddens, masks, actions):
        value, x, auxiliary, hiddens = self(inputs, hiddens, masks)

        slices = torch.split(x, donkey.CONTROL_SIZE, 1)
        action_mean = slices[0]
        action_logstd = slices[1]
        action_std = action_logstd.exp()

        if self.config.get('fixed_action_std'):
            action_std = (
                self.config.get('fixed_action_std') *
                torch.ones(action_mean.size()).float()
            )
            if self.config.get('cuda'):
                action_std = action_std.cuda()
            action_std = autograd.Variable(action_std)
            action_logstd = action_std.log()

        m = Normal(action_mean, action_std)

        # log_probs (sum on actions) -> batch x 1
        log_probs = m.log_prob(actions).sum(-1, keepdim=True)

        # entropy (sum on actions / mean on batch) -> 1x1
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        entropy = entropy.sum(-1, keepdim=True)

        return value, auxiliary, hiddens, log_probs, entropy
