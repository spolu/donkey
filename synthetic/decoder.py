import math
import cv2
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Categorical

from synthetic import State

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.device = torch.device(config.get('device'))

        self.fc1 = nn.Linear(State.size(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5_mean = nn.Linear(256, 256)
        self.fc5_logvar = nn.Linear(256, 256)

        self.dcv1 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.dcv2 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.dcv3 = nn.ConvTranspose2d(32, 16, 4, stride=2)
        self.dcv4 = nn.ConvTranspose2d(16, 8, 4, stride=2)
        self.dcv5 = nn.ConvTranspose2d(8, 4, 5, stride=2)
        self.dcv6 = nn.ConvTranspose2d(4, 1, 5, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0)
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, nn.init.calculate_gain('relu'))
                m.bias.data.fill_(0)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, state, deterministic=False):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        mean, logvar = self.fc5_mean(x), self.fc5_logvar(x)

        z = mean
        if not deterministic:
            z = self.reparameterize(mean, logvar)

        z = z.view(-1, 128, 1, 2)

        # print("z {}".format(z.size()))
        z = F.relu(self.dcv1(z))
        # print("dcv1 {}".format(z.size()))
        z = F.relu(self.dcv2(z))
        # print("dcv2 {}".format(z.size()))
        z = F.relu(self.dcv3(z))
        # print("dcv3 {}".format(z.size()))
        z = F.relu(self.dcv4(z))
        # print("dcv4 {}".format(z.size()))
        z = F.relu(self.dcv5(z))
        # print("dcv5 {}".format(z.size()))
        z = F.sigmoid(self.dcv6(z))
        # print("dcv6 {}".format(z.size()))

        z = z[:,:,35:105,20:180].squeeze(1)

        return z, mean, logvar
