import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class WMVAE(nn.Module):
    def __init__(self, config):
        super(WMVAE, self).__init__()

        self.z_size = config.get('world_model_z_size')

        # Encoder.
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        # Latent representation of mean and std.
        self.fc1 = nn.Linear(256 * 6 * 6, self.z_size)
        self.fc2 = nn.Linear(256 * 6 * 6, self.z_size)
        self.fc3 = nn.Linear(self.z_size, 256 * 6 * 6)

        # Decoder.
        self.deconv1 = nn.ConvTranspose2d(256 * 6 *6, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 16, 6, stride=2)
        self.deconv5 = nn.ConvTranspose2d(16, 3, 6, stride=2)

    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(-1, 256 * 6 * 6)
        return self.fc1(h), self.fc2(h)

    def decode(self, z):
        h = self.fc3(z).view(-1, 256 * 6 * 6, 1, 1)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        h = F.relu(self.deconv4(h))
        h = F.sigmoid(self.deconv5(h))
        return h

    def forward(self, x, encode_only=False, deterministic=True):
        mu, logvar = self.encode(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        if encode_only:
            if deterministic:
                return mu
            return z

        return self.decode(z), mu, logvar

