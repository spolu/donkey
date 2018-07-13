import math
import cv2
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Categorical

import reinforce
from reinforce.input_filter import InputFilter

# import pdb; pdb.set_trace()

CONV_OUT_WIDTH = 2
CONV_OUT_HEIGHT = 8

class VAECroppedEdges(nn.Module):
    def __init__(self, config):
        super(VAECroppedEdges, self).__init__()
        self.latent_size = config.get('latent_size')

        self.device = torch.device(config.get('device'))

        self.input_filter = InputFilter(config)

        ## Encoder
        self.cv1 = nn.Conv2d(1, 32, 4, stride=2)
        self.cv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.cv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.cv4 = nn.Conv2d(128, 256, 4, stride=2)

        ## Latent representation of mean and std
        self.fc1 = nn.Linear(256*CONV_OUT_WIDTH*CONV_OUT_HEIGHT, self.latent_size)
        self.fc2 = nn.Linear(256*CONV_OUT_WIDTH*CONV_OUT_HEIGHT, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size, 256*CONV_OUT_WIDTH*CONV_OUT_HEIGHT)

        ## Decoder
        self.dcv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, output_padding=(1,0))
        self.dcv2 = nn.ConvTranspose2d(128, 64, 4, stride=2)
        self.dcv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, output_padding=(0,1))
        self.dcv4 = nn.ConvTranspose2d(32, 1, 5, stride=2, output_padding=1, padding=1)

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

        return self.fc1(x), self.fc2(x)

    def decode(self, z):
        x = self.fc3(z)

        x = x.view(-1, 256, CONV_OUT_WIDTH, CONV_OUT_HEIGHT)

        x = F.relu(self.dcv1(x))
        # print("dcv1 {}".format(x.size()))
        x = F.relu(self.dcv2(x))
        # print("dcv2 {}".format(x.size()))
        x = F.relu(self.dcv3(x))
        # print("dcv3 {}".format(x.size()))
        x = F.sigmoid(self.dcv4(x))
        # print("dcv4 {}".format(x.size()))

        # import pdb; pdb.set_trace()

        return x

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, x, encode=False, deterministic=False):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)

        if deterministic:
            z = mean

        if encode:
            return z
        else:
            reconstruct = self.decode(z)
            return reconstruct, mean, logvar

    def input_shape(self):
        return (
            1,
            int(reinforce.CAMERA_HEIGHT - 50),
            int(reinforce.CAMERA_WIDTH),
        )

    def input(self, observation):
        cameras = [
            self.input_filter.apply(o.camera_raw) / 255.0
            for o in observation
        ]

        observation = np.concatenate(
            (
                np.stack(cameras),
            ),
            axis=-1,
        )
        observation = torch.from_numpy(observation).float().unsqueeze(1)

        return observation

