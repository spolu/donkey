import math
import cv2

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.distributions import Normal, Categorical

import reinforce

class PPOPixelsCNNCroppedEdges(nn.Module):
    def __init__(self, config):
        super(PPOPixelsCNNCroppedEdges, self).__init__()
        self.hidden_size = config.get('hidden_size')
        self.recurring_cell = config.get('recurring_cell')
        self.action_type = config.get('action_type')
        self.config = config

        self.cv1 = nn.Conv2d(reinforce.CAMERA_STACK_SIZE, 24, 5, stride=2)
        self.cv2 = nn.Conv2d(24, 32, 5, stride=2)
        self.cv3 = nn.Conv2d(32, 64, 3, stride=2)
        self.cv4 = nn.Conv2d(64, 64, 3, stride=1)
        self.cv5 = nn.Conv2d(64, 64, 3, stride=1)

        self.dp1 = nn.Dropout(p=0.1)

        self.fc1 = nn.Linear(234, self.hidden_size)

        if self.recurring_cell == "gru":
            self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)

        self.fc1_a = nn.Linear(self.hidden_size, self.hidden_size)
        if self.action_type == 'discrete':
            self.fc2_a = nn.Linear(self.hidden_size, reinforce.DISCRETE_CONTROL_SIZE)
        else:
            self.fc2_a = nn.Linear(self.hidden_size, 2 * reinforce.CONTINUOUS_CONTROL_SIZE)

        self.fc1_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_v = nn.Linear(self.hidden_size, 1)

        self.train()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc1.weight.data, nn.init.calculate_gain('relu'))
        self.fc1.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc1_a.weight.data, nn.init.calculate_gain('tanh'))
        if self.action_type == 'discrete':
            nn.init.xavier_normal(self.fc2_a.weight.data, nn.init.calculate_gain('linear'))
        else:
            nn.init.xavier_normal(self.fc2_a.weight.data, nn.init.calculate_gain('tanh'))
        self.fc1_a.bias.data.fill_(0)
        self.fc2_a.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc1_v.weight.data, nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal(self.fc2_v.weight.data, nn.init.calculate_gain('linear'))
        self.fc1_v.bias.data.fill_(0)
        self.fc2_v.bias.data.fill_(0)

        if self.recurring_cell == "gru":
            nn.init.xavier_normal(self.gru.weight_ih.data)
            nn.init.xavier_normal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    def forward(self, inputs, hiddens, masks):
        x = F.elu(self.cv1(inputs))
        x = F.elu(self.cv2(x))
        x = F.elu(self.cv3(x))
        x = F.elu(self.cv4(x))
        x = F.elu(self.cv5(x))
        x = self.dp1(x)

        x = x.view(-1, 234)
        x = F.elu(self.fc1(x))

        if self.recurring_cell == "gru":
            if inputs.size(0) == hiddens.size(0):
                x = hiddens = self.gru(x, hiddens * masks)
            else:
                x = x.view(-1, hiddens.size(0), x.size(1))
                masks = masks.view(-1, hiddens.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = hiddens = self.gru(x[i], hiddens * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)
            x = F.tanh(x)

        # Action head.
        a = F.tanh(self.fc1_a(x))
        if self.action_type == 'discrete':
            a = self.fc2_a(a)
        else:
            a = F.tanh(self.fc2_a(a))

        # Value head.
        v = F.tanh(self.fc1_v(x))
        v = self.fc2_v(v)

        return v, a, hiddens

    def input_shape(self):
        return (
            1,
            reinforce.CAMERA_HEIGHT-50,
            reinforce.CAMERA_WIDTH,
        )

    def input(self, observation):
        cameras = [
            cv2.Canny(
                o.camera.astype(np.uint8), 50, 150, apertureSize = 3,
            ).astype(np.float)[50:]
            for o in observation
        ]

        observation = np.concatenate(
            (
                np.stack(cameras),
            ),
            axis=-1,
        )
        observation = torch.from_numpy(observation).float()

        return observation

    def action(self, inputs, hiddens, masks, deterministic=False):
        value, x, auxiliary, hiddens = self(inputs, hiddens, masks)

        if self.action_type == 'discrete':
            probs = F.softmax(x, dim=1)
            log_probs = F.log_softmax(x, dim=1)

            m = Categorical(probs)
            actions = m.sample().view(-1, 1)

            action_log_probs = log_probs.gather(1, actions)
            entropy = -(log_probs * probs).sum(-1).mean()

            return value, actions, auxiliary, hiddens, action_log_probs, entropy
        else:
            slices = torch.split(x, reinforce.CONTINUOUS_CONTROL_SIZE, 1)
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

        if self.action_type == 'discrete':
            probs = F.softmax(x, dim=1)
            log_probs = F.log_softmax(x, dim=1)

            m = Categorical(probs)
            actions = m.sample().view(-1, 1)

            action_log_probs = log_probs.gather(1, actions)
            entropy = -(log_probs * probs).sum(-1).mean()

            return value, auxiliary, hiddens, action_log_probs, entropy
        else:
            slices = torch.split(x, reinforce.CONTINUOUS_CONTROL_SIZE, 1)
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