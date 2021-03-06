import math

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

# import pdb; pdb.set_trace()

class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        self.config = config

        self.relu = nn.ReLU()

        self.cv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, bias=True)
        # self.bn1 = nn.BatchNorm2d(24)
        self.cv2 = nn.Conv2d(24, 32, kernel_size=5, stride=2, bias=True)
        self.cv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, bias=True)
        self.cv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=True)
        self.cv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=True)

        self.fc1 = nn.Linear(8064, 128)
        self.fc2 = nn.Linear(128, 64)

        self.dp1 = nn.Dropout(p=0.3)

        self.fc_v1 = nn.Linear(64, 2)
        self.fc_v2 = nn.Linear(64, 1)

        nn.init.xavier_normal_(self.fc1.weight.data, nn.init.calculate_gain('relu'))
        self.fc1.bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc2.weight.data, nn.init.calculate_gain('relu'))
        self.fc2.bias.data.fill_(0)

        nn.init.xavier_normal_(self.fc_v1.weight.data, nn.init.calculate_gain('tanh'))
        self.fc_v1.bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc_v2.weight.data, nn.init.calculate_gain('linear'))
        self.fc_v2.bias.data.fill_(0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        x = self.cv1(inputs)
        # x = self.bn1(x)
        x = self.relu(x)

        x = self.cv2(x)
        x = self.relu(x)

        x = self.cv3(x)
        x = self.relu(x)

        x = self.cv4(x)
        x = self.relu(x)

        x = self.cv5(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.dp1(x)

        return torch.cat(
            (F.tanh(self.fc_v1(x)), self.fc_v2(x)), 1,
        )
