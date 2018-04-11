import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.distributions import Normal, Categorical

import donkey

class Policy(nn.Module):
    def __init__(self, config):
        super(Policy, self).__init__()
        self.hidden_size = config.get('hidden_size')
        self.recurring_cell = config.get('recurring_cell')
        self.action_type = config.get('action_type')
        self.config = config

        self.cv1 = nn.Conv2d(donkey.CAMERA_STACK_SIZE, 24, 5, stride=2)
        self.cv2 = nn.Conv2d(24, 32, 5, stride=2)
        self.cv3 = nn.Conv2d(32, 64, 5, stride=2)
        self.cv4 = nn.Conv2d(64, 64, 3, stride=2)
        self.cv5 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(1152, self.hidden_size)

        if self.recurring_cell == "gru":
            self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)

        self.fc1_x = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_x = nn.Linear(self.hidden_size, donkey.ANGLES_WINDOW)

        self.fc1_a = nn.Linear(self.hidden_size, self.hidden_size)
        if self.action_type == 'discrete':
            self.fc2_a = nn.Linear(self.hidden_size, donkey.DISCRETE_CONTROL_SIZE)
        else:
            self.fc2_a = nn.Linear(self.hidden_size, 2 * donkey.CONTINUOUS_CONTROL_SIZE)

        self.fc1_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_v = nn.Linear(self.hidden_size, 1)

        self.train()

        nn.init.xavier_normal(self.cv1.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv2.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv3.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv4.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv5.weight.data, nn.init.calculate_gain('relu'))
        self.cv1.bias.data.fill_(0)
        self.cv2.bias.data.fill_(0)
        self.cv3.bias.data.fill_(0)
        self.cv4.bias.data.fill_(0)
        self.cv5.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc1.weight.data, nn.init.calculate_gain('relu'))
        self.fc1.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc1_x.weight.data, nn.init.calculate_gain('tanh'))
        self.fc1_x.bias.data.fill_(0)
        nn.init.xavier_normal(self.fc2_x.weight.data, nn.init.calculate_gain('linear'))
        self.fc2_x.bias.data.fill_(0)

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

        x = x.view(-1, 1152)

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

        # Auxiliary head.
        ax = F.tanh(self.fc1_x(x))
        ax = self.fc2_x(ax)

        # Value head.
        v = F.tanh(self.fc1_v(x))
        v = self.fc2_v(v)

        return v, a, ax, hiddens

    def input_shape(self):
        return (
            donkey.CAMERA_STACK_SIZE,
            donkey.CAMERA_WIDTH,
            donkey.CAMERA_HEIGHT,
        )

    def input(self, observation):
        cameras = [o.camera for o in observation]
        observation = np.concatenate(
            (
                np.stack(cameras),
            ),
            axis=-1,
        )
        observation = torch.from_numpy(observation).float()

        return observation

    def auxiliary_present(self):
        return True

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
            slices = torch.split(x, donkey.CONTINUOUS_CONTROL_SIZE, 1)
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
            slices = torch.split(x, donkey.CONTINUOUS_CONTROL_SIZE, 1)
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
