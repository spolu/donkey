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

CONV_OUT_WIDTH = 2
CONV_OUT_HEIGHT = 8

class VAE(nn.Module):
    def __init__(self,config):
        super(VAE, self).__init__()
        self.device = torch.device(config.get('device'))
        self.latent_size = config.get('latent_size')

        ## Encoder
        self.cv1 = nn.Conv2d(1, 32, 4, stride=2)
        self.cv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.cv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.cv4 = nn.Conv2d(128, 256, 4, stride=2)

        ## Latent representation of mean and std
        self.fc_mean = nn.Linear(256*CONV_OUT_WIDTH*CONV_OUT_HEIGHT, self.latent_size)
        self.fc_logvar = nn.Linear(256*CONV_OUT_WIDTH*CONV_OUT_HEIGHT, self.latent_size)
        self.fc_latent = nn.Linear(self.latent_size, 256*CONV_OUT_WIDTH*CONV_OUT_HEIGHT)

        ## Decoder
        self.dcv1 = nn.ConvTranspose2d(256, 128, 5, stride=2)
        self.dcv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.dcv3 = nn.ConvTranspose2d(64, 32, 5, stride=2)
        self.dcv4 = nn.ConvTranspose2d(32, 1, 5, stride=2)

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

    def encode(self, inputs):
        x = F.relu(self.cv1(inputs))
        # print("cv1 {}".format(x.size()))
        x = F.relu(self.cv2(x))
        # print("cv2 {}".format(x.size()))
        x = F.relu(self.cv3(x))
        # print("cv3 {}".format(x.size()))
        x = F.relu(self.cv4(x))
        # print("cv4 {}".format(x.size()))

        # import pdb; pdb.set_trace()
        x = x.view(-1, 256*CONV_OUT_WIDTH*CONV_OUT_HEIGHT)

        return self.fc_mean(x), self.fc_logvar(x)

    def decode(self, z):
        x = self.fc_latent(z)

        x = x.view(-1, 256, CONV_OUT_WIDTH, CONV_OUT_HEIGHT)

        x = F.relu(self.dcv1(x))
        # print("dcv1 {}".format(x.size()))
        x = F.relu(self.dcv2(x))
        # print("dcv2 {}".format(x.size()))
        x = F.relu(self.dcv3(x))
        # print("dcv3 {}".format(x.size()))
        x = F.sigmoid(self.dcv4(x))
        # print("dcv4 {}".format(x.size()))

        padh = int((x.size(2) - OUT_HEIGHT) / 2)
        padw = int((x.size(3) - OUT_WIDTH) / 2)

        x = x[:,:,padh:OUT_HEIGHT+padh,padw:OUT_WIDTH+padw]
        # import pdb; pdb.set_trace()

        return x

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, x, deterministic=False):
        mean, logvar = self.encode(x)
        latent = self.reparameterize(mean, logvar)

        if deterministic:
            latent = mean

        encoded = self.decode(latent)

        return latent, encoded, mean, logvar

class STL(nn.Module):
    def __init__(self, config):
        super(STL, self).__init__()
        self.device = torch.device(config.get('device'))
        self.latent_size = config.get('latent_size')

        self.fc1 = nn.Linear(State.size(), 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc_mean = nn.Linear(256, self.latent_size)
        self.fc_logvar = nn.Linear(256, self.latent_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, nn.init.calculate_gain('linear'))
                m.bias.data.fill_(0)

    def encode(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.fc_mean(x), self.fc_logvar(x)

    def forward(self, state, deterministic=False):
        mean, logvar = self.encode(state)
        latent = self.reparameterize(mean, logvar)

        if deterministic:
            latent = mean

        return latent, mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean
