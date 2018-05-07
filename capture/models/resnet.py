import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.distributions import Normal, Categorical

class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.cv1_rs = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False,
        )
        self.bn1_rs = nn.BatchNorm2d(planes)
        self.cv2_rs = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1,
            bias=False,
        )
        self.bn2_rs = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.cv1_rs(x)
        out = self.bn1_rs(out)
        out = F.relu(out, inplace=True)

        out = self.cv2_rs(out)
        out = self.bn2_rs(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        return out

class HeadBlock(nn.Module):
    def __init__(self, hidden_size, nonlinearity, output_size, stride=1):
        super(HeadBlock, self).__init__()
        self.fc1_hd = nn.Linear(hidden_size, hidden_size)
        self.fc2_hd = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal_(self.fc1_hd.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc2_hd.weight.data, nn.init.calculate_gain(nonlinearity))
        self.fc1_hd.bias.data.fill_(0)
        self.fc2_hd.bias.data.fill_(0)

    def forward(self, x):
        out = self.fc1_hd(x)
        out = F.relu(out, inplace=True)
        out = self.fc2_hd(out)

        return out


class ResNet(nn.Module):
    def __init__(self, config, value_count):
        super(ResNet, self).__init__()
        self.hidden_size = config.get('hidden_size')
        self.config = config
        self.inplanes = 64
        self.value_count = value_count

        self.cv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rs1 = self._make_residual_layer(64, 1)
        self.rs2 = self._make_residual_layer(128, 1)
        self.rs3 = self._make_residual_layer(256, 1)

        self.cv2 = nn.Conv2d(
            256, 1, kernel_size=1, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(32*42, self.hidden_size)

        # Value head.
        self.hd_v = HeadBlock(self.hidden_size, 'linear', self.value_count)

        nn.init.xavier_normal_(self.fc1.weight.data, nn.init.calculate_gain('relu'))
        self.fc1.bias.data.fill_(0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def forward(self, inputs):
        x = self.cv1(inputs)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        x = self.rs1(x)
        x = self.rs2(x)
        x = self.rs3(x)

        x = self.cv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)

        return self.hd_v(x)
