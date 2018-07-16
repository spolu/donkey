import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic import State

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.device = torch.device(config.get('device'))

        self.cv1 = nn.Conv2d(1, 64, 4, stride=2, padding=1)

        self.cv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)
        self.cv_bn2 = nn.BatchNorm2d(128)

        self.cv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)
        self.cv_bn3 = nn.BatchNorm2d(256)

        self.cv4 = nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)
        self.cv_bn4 = nn.BatchNorm2d(512)

        self.cv5 = nn.Conv2d(512, 1, 4, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, pair):
        # print("x {}".format(pair.size()))
        x = F.leaky_relu(self.cv1(pair), 0.2)
        # print("cv1 x {}".format(x.size()))
        x = F.leaky_relu(self.cv_bn2(self.cv2(x)), 0.2)
        # print("cv2 x {}".format(x.size()))
        x = F.leaky_relu(self.cv_bn3(self.cv3(x)), 0.2)
        # print("cv3 x {}".format(x.size()))
        x = F.leaky_relu(self.cv_bn4(self.cv4(x)), 0.2)
        # print("cv4 x {}".format(x.size()))
        x = F.sigmoid(self.cv5(x))
        # print("cv5 x {}".format(x.size()))

        return x

    def set_requires_grad(self, requires_grad=True):
        for p in self.parameters():
            p.requires_grad = requires_grad
