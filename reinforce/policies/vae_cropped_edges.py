import math
import cv2
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Categorical

import reinforce

# import pdb; pdb.set_trace()

CONV_OUT_WIDTH = 15
CONV_OUT_HEIGHT = 37

class VAECroppedEdges(nn.Module):
    def __init__(self, config):
        super(VAECroppedEdges, self).__init__()
        self.latent_size = config.get('latent_size')

        self.device = torch.device(config.get('device'))

        ## Encoder
        self.cv1 = nn.Conv2d(1, 24, 5, stride=2)
        self.cv2 = nn.Conv2d(24, 32, 5, stride=2)

        ## Latent representation of mean and std
        self.fc1 = nn.Linear(32*CONV_OUT_WIDTH*CONV_OUT_HEIGHT, self.latent_size)
        self.fc2 = nn.Linear(32*CONV_OUT_WIDTH*CONV_OUT_HEIGHT, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size, 32*CONV_OUT_WIDTH*CONV_OUT_HEIGHT)

        ## Decoder
        self.dcv1 = nn.ConvTranspose2d(32, 24, (5, 6), stride=2)
        self.dcv2 = nn.ConvTranspose2d(24, 1, 6, stride=2)

    def forward(self, inputs):
        x = F.relu(self.cv1(inputs))
        x = F.relu(self.cv2(x))

        # import pdb; pdb.set_trace()
        x = x.view(-1, 32*CONV_OUT_WIDTH*CONV_OUT_HEIGHT)

        return self.fc1(x), self.fc2(x)

    def reconstruct(self, z):
        x = self.fc3(z).view(-1, 32, CONV_OUT_WIDTH, CONV_OUT_HEIGHT)
        x = F.relu(self.dcv1(x))
        x = F.relu(self.dcv2(x))
        x = F.sigmoid(x)

        return x

    def encode(self, inputs):
        mu, _ = self.forward(inputs)
        return mu

    def decode(self, inputs):
        mu, logvar = self.forward(inputs)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        return self.reconstruct(z), mu, logvar

    def input_shape(self):
        return (
            1,
            int(reinforce.CAMERA_HEIGHT - 50),
            int(reinforce.CAMERA_WIDTH),
        )

    def input(self, observation):
        cameras = [
            cv2.Canny(
                cv2.imdecode(
                    np.fromstring(o.camera_raw, np.uint8),
                    cv2.IMREAD_GRAYSCALE,
                ),
                50, 150, apertureSize = 3,
            )[50:] / 127.5 - 1
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

