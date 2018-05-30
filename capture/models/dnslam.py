import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.distributions import Normal, Categorical

class ResidualBlockNoBatch(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlockNoBatch, self).__init__()
        self.cv1_rs = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.cv2_rs = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.cv1_rs(x)
        out = F.relu(out, inplace=True)
        out = self.cv2_rs(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        return out

class DNSLAM(nn.Module):
    def __init__(self, config, stack_count):
        super(DNSLAM, self).__init__()
        self.hidden_size = config.get('hidden_size')
        self.config = config
        self.inplanes = 64

        self.stack_count = stack_count

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.cv1 = nn.Conv2d(
            self.stack_count, 64,
            kernel_size=7, stride=2, padding=3, bias=False,
        )
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rs11 = self._make_residual_layer(64, 1, stride=1)
        self.rs12 = self._make_residual_layer(64, 1, stride=2)
        self.rs21 = self._make_residual_layer(128, 1, stride=1)
        self.rs22 = self._make_residual_layer(128, 1, stride=2)

        self.avgp = nn.AvgPool2d(8, stride=1)

        self.fc1 = nn.Linear(3*128, self.hidden_size)
        self.rnn = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.1)

        self.fc_progress = nn.Linear(self.hidden_size, 1)
        self.fc_position = nn.Linear(self.hidden_size, 1)
        self.fc_angle = nn.Linear(self.hidden_size, 1)
        self.fc_speed = nn.Linear(self.hidden_size, 1)

        # Initialization
        nn.init.xavier_normal_(self.fc1.weight.data, nn.init.calculate_gain('relu'))
        self.fc1.bias.data.fill_(0)

        nn.init.xavier_normal_(self.fc_progress.weight.data, nn.init.calculate_gain('sigmoid'))
        self.fc_progress.bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc_position.weight.data, nn.init.calculate_gain('tanh'))
        self.fc_position.bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc_angle.weight.data, nn.init.calculate_gain('tanh'))
        self.fc_angle.bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc_speed.weight.data, nn.init.calculate_gain('sigmoid'))
        self.fc_speed.bias.data.fill_(0)

        nn.init.xavier_normal_(self.rnn.weight_ih.data)
        nn.init.xavier_normal_(self.rnn.weight_hh.data)
        self.rnn.bias_ih.data.fill_(0)
        self.rnn.bias_hh.data.fill_(0)

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
        layers.append(ResidualBlockNoBatch(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(ResidualBlockNoBatch(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, inputs, hiddens):
        x = self.cv1(inputs)
        x = self.relu(x)
        x = self.maxp(x)

        x = self.rs11(x)
        x = self.rs12(x)
        x = self.rs21(x)
        x = self.rs22(x)

        x = self.avgp(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)

        x = h = self.rnn(x, hiddens)
        x = self.relu(x)
        x = self.dropout(x)

        progress = self.sigmoid(self.fc_progress(x))
        position = self.tanh(self.fc_position(x))
        angle = self.tanh(self.fc_angle(x))
        speed = self.sigmoid(self.fc_speed(x))

        return progress, position, angle, speed, h
