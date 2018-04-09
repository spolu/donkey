import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.distributions import Normal, Categorical

import donkey

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Policy(nn.Module):
    def __init__(self, config):
        super(Policy, self).__init__()
        self.hidden_size = config.get('hidden_size')
        self.action_type = config.get('action_type')
        self.config = config
        self.gradients = None
        self.inplanes = 64

        self.cv1 = nn.Conv2d(
            donkey.CAMERA_STACK_SIZE, 64, kernel_size=7, stride=2, padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rs1 = self._make_layer(64, 3)
        self.rs2 = self._make_layer(128, 4)
        self.rs3 = self._make_layer(256, 6)
        self.rs4 = self._make_layer(512, 3)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(417792, self.hidden_size)

        # Action network.
        self.ax1_a = nn.Linear(self.hidden_size, donkey.ANGLES_WINDOW)

        self.fc1_a = nn.Linear(self.hidden_size, self.hidden_size)
        if self.action_type == 'discrete':
            self.fc2_a = nn.Linear(self.hidden_size, donkey.DISCRETE_CONTROL_SIZE)
        else:
            self.fc2_a = nn.Linear(self.hidden_size, 2 * donkey.CONTINUOUS_CONTROL_SIZE)

        # Value network.
        self.fc1_v = nn.Linear(self.hidden_size + donkey.ANGLES_WINDOW, self.hidden_size)
        self.fc2_v = nn.Linear(self.hidden_size, 1)

        self.train()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.xavier_normal(self.fc.weight.data, nn.init.calculate_gain('relu'))
        self.fc.bias.data.fill_(0)

        nn.init.xavier_normal(self.ax1_a.weight.data, nn.init.calculate_gain('linear'))
        self.ax1_a.bias.data.fill_(0)

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

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes, kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        self.cv1.register_backward_hook(hook_function)

    def forward(self, inputs, hiddens, masks):
        # A little bit of pytorch magic to extract camera pixels and angles
        # from the packed input.
        pixels_inputs, stack = torch.split(
            inputs, (donkey.CAMERA_STACK_SIZE), dim=1
        )
        angles_inputs = stack.split(1, 2)[0].split(
            (donkey.ANGLES_WINDOW), 3,
        )[0].contiguous().view(-1, donkey.ANGLES_WINDOW)


        x = self.cv1(pixels_inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.rs1(x)
        x = self.rs2(x)
        x = self.rs3(x)
        x = self.rs4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = F.elu(self.fc(x))

        # Action network.
        angles = self.ax1_a(x)

        a = F.tanh(self.fc1_a(x))
        if self.action_type == 'discrete':
            a = self.fc2_a(a)
        else:
            a = F.tanh(self.fc2_a(a))

        # Value network.
        v = F.tanh(self.fc1_v(torch.cat((x, angles_inputs), 1)))
        v = self.fc2_v(v)

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
