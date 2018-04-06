import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.distributions import Normal

class Policy(nn.Module):
    def __init__(self, config):
        super(Policy, self).__init__()
        self.hidden_size = 256
        self.recurring_cell = 'none'

        self.cv1 = nn.Conv2d(4, 24, 5, stride=2)
        self.cv2 = nn.Conv2d(24, 32, 5, stride=2)
        self.cv3 = nn.Conv2d(32, 64, 5, stride=2)
        self.cv4 = nn.Conv2d(64, 64, 3, stride=2)
        self.cv5 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(1152, self.hidden_size)

        if self.recurring_cell == "gru":
            self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)

        self.ax1_a = nn.Linear(self.hidden_size, 8)

        self.fc1_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_a = nn.Linear(self.hidden_size, 2 * 2)

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

        nn.init.xavier_normal(self.ax1_a.weight.data, nn.init.calculate_gain('linear'))
        self.ax1_a.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc1_a.weight.data, nn.init.calculate_gain('tanh'))
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

        angles = self.ax1_a(x)

        a = F.tanh(self.fc1_a(x))
        a = F.tanh(self.fc2_a(a))

        v = F.tanh(self.fc1_v(x))
        v = self.fc2_v(v)

        return v, a, angles, hiddens

    def input_shape(self):
        return (4, 120, 160)

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
        return (8,)

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

    def action(self, inputs, hiddens, masks):
        value, x, auxiliary, hiddens = self(inputs, hiddens, masks)

        slices = torch.split(x, donkey.CONTROL_SIZE, 1)
        action_mean = slices[0]
        action_logstd = slices[1]
        action_std = action_logstd.exp()

        actions = action_mean

        # log_probs (sum on actions) -> batch x 1
        log_probs = m.log_prob(actions).sum(-1, keepdim=True)

        # entropy (sum on actions / mean on batch) -> 1x1
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        entropy = entropy.sum(-1, keepdim=True)

        return value, actions, auxiliary, hiddens, log_probs, entropy

class Runner:
    def __init__(self, load_dir):
        self.load_dir = load_dir

        self.policy = Policy()
        self.policy.load_state_dict(
            torch.load(self.load_dir + "/policy.pt", map_location='cpu'),
        )
        self.policy.eval()

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        observation = torch.from_numpy(img_arr).float()

        value, x, auxiliary, hiddens = self.policy(
            autograd.Variable(
                observation, requires_grad=False,
            ),
            autograd.Variable(
                torch.zeros(1, 256), requires_grad=False,
            ),
            autograd.Variable(
                torch.ones(1), requires_grad=False,
            ),
        )

        slices = torch.split(x, 2, 1)
        actions = slices[0].data.numpy()

        return actions[0], actions[1]

    def shutdown(self):
        pass
