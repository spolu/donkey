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
        self.fc2_mean = nn.Linear(256, 256)
        self.fc2_logvar = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256*1*2)

        self.dcv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, output_padding=(0,1))
        self.dcv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, output_padding=(0,1))
        self.dcv3 = nn.ConvTranspose2d(64, 32, 4, stride=(1,2), output_padding=(0,1))
        self.dcv4 = nn.ConvTranspose2d(32, 32, 4, stride=(1,2))
        self.dcv5 = nn.ConvTranspose2d(32, 16, 4, stride=(2,1))
        self.dcv6 = nn.ConvTranspose2d(16, 1, 4, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, nn.init.calculate_gain('linear'))
                m.bias.data.fill_(0)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, state, deterministic=False):
        x = F.relu(self.fc1(state))
        mean, logvar = self.fc2_mean(x), self.fc2_logvar(x)

        z = mean
        if not deterministic:
            z = self.reparameterize(mean, logvar)

        z = F.relu(self.fc3(z))
        z = z.view(-1, 256, 1, 2)

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

        z = z.squeeze(1)

        return z, mean, logvar
