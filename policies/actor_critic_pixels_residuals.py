import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.distributions import Normal, Categorical

import donkey

class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        return out

class HeadBlock(nn.Module):
    def __init__(self, inplanes, hidden_size, nonlinearity, output_size, stride=1):
        super(HeadBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, 1, kernel_size=1, stride=stride, padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(32*42, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal(self.fc1.weight.data, nn.init.calculate_gain('relu'))
        self.fc1.bias.data.fill_(0)
        nn.init.xavier_normal(self.fc2.weight.data, nn.init.calculate_gain(nonlinearity))
        self.fc2.bias.data.fill_(0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out, inplace=True)

        out = self.fc2(out)

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
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rs1 = self._make_residual_layer(64, 2)
        self.rs2 = self._make_residual_layer(128, 2)
        self.rs3 = self._make_residual_layer(256, 1)

        # Action head.
        if self.action_type == 'discrete':
            self.hd_a = HeadBlock(
                256, 256, 'linear', donkey.DISCRETE_CONTROL_SIZE,
            )
        else:
            self.hd_a = HeadBlock(
                256, 256, 'tanh', 2*donkey.CONTINUOUS_CONTROL_SIZE,
            )
        # Auxiliary head.
        self.hd_x = HeadBlock(256, 256, 'linear', donkey.ANGLES_WINDOW)
        # Value head.
        self.hd_v = HeadBlock(256, 256, 'linear', 1)

        self.train()

    def _make_residual_layer(self, planes, blocks, stride=1):
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
        layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        self.cv1.register_backward_hook(hook_function)

    def forward(self, inputs, hiddens, masks):
        x = self.cv1(inputs)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        x = self.rs1(x)
        x = self.rs2(x)
        x = self.rs3(x)

        if self.action_type == 'discrete':
            a = self.hd_a(x)
        else:
            a = F.tanh(self.hd_a(x))
        ax = self.hd_x(x)
        v = self.hd_v(x)

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
