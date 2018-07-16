import math
import cv2
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic import State

# import pdb; pdb.set_trace()

OUT_HEIGHT = 70
OUT_WIDTH = 160

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.device = torch.device(config.get('device'))

        self.fc1 = nn.Linear(State.size(), 256, bias=False)
        self.fc_bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 256, bias=False)
        self.fc_bn2 = nn.BatchNorm1d(256)

        self.fc_mean = nn.Linear(256, 256)
        self.fc_logvar = nn.Linear(256, 256)

        self.dcv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=0, bias=False)
        self.dcv_bn1 = nn.BatchNorm2d(64)

        self.dcv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)
        self.dcv_bn2 = nn.BatchNorm2d(32)

        self.dcv3 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False)
        self.dcv_bn3 = nn.BatchNorm2d(16)

        self.dcv4 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, bias=False)
        self.dcv_bn4 = nn.BatchNorm2d(8)

        self.dcv5 = nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1, bias=False)
        self.dcv_bn5 = nn.BatchNorm2d(4)

        self.dcv6 = nn.ConvTranspose2d(4, 1, 4, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data, nn.init.calculate_gain('linear'))
                    m.bias.data.fill_(0)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, state, deterministic=False):
        x = F.relu(self.fc_bn1(self.fc1(state)))
        x = F.relu(self.fc_bn2(self.fc2(x)))
        mean, logvar = self.fc_mean(x), self.fc_logvar(x)

        z = mean
        if not deterministic:
            z = self.reparameterize(mean, logvar)

        z = z.view(-1, 128, 1, 2)

        # print("z {}".format(z.size()))
        z = F.relu(self.dcv_bn1(self.dcv1(z)))
        # print("dcv1 {}".format(z.size()))
        z = F.relu(self.dcv_bn2(self.dcv2(z)))
        # print("dcv2 {}".format(z.size()))
        z = F.relu(self.dcv_bn3(self.dcv3(z)))
        # print("dcv3 {}".format(z.size()))
        z = F.relu(self.dcv_bn4(self.dcv4(z)))
        # print("dcv4 {}".format(z.size()))
        z = F.relu(self.dcv_bn5(self.dcv5(z)))
        # print("dcv5 {}".format(z.size()))
        z = F.sigmoid(self.dcv6(z))
        # print("dcv6 {}".format(z.size()))

        padh = int((z.size(2) - OUT_HEIGHT) / 2)
        padw = int((z.size(3) - OUT_WIDTH) / 2)

        z = z[:,:,padh:OUT_HEIGHT+padh,padw:OUT_WIDTH+padw]

        return z, mean, logvar
