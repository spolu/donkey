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

CONV_OUT_WIDTH = 1
CONV_OUT_HEIGHT = 1

class VAE(nn.Module):
    def __init__(self,config):
        super(VAE, self).__init__()
        self.device = torch.device(config.get('device'))
        self.stl_latent_size = config.get('stl_latent_size')
        self.frame_stack_size = config.get('frame_stack_size')

        ## Encoder
        self.cv1 = nn.Conv2d(self.frame_stack_size, 16, 5, stride=2, bias=False)
        self.bn_cv1 = nn.BatchNorm2d(16)
        self.cv2 = nn.Conv2d(16, 32, 4, stride=(1,2), bias=False)
        self.bn_cv2 = nn.BatchNorm2d(32)
        self.cv3 = nn.Conv2d(32, 64, 4, stride=2, bias=False)
        self.bn_cv3 = nn.BatchNorm2d(64)
        self.cv4 = nn.Conv2d(64, 128, 4, stride=2, bias=False)
        self.bn_cv4 = nn.BatchNorm2d(128)
        self.cv5 = nn.Conv2d(128, 256, 4, stride=2, bias=False)
        self.bn_cv5 = nn.BatchNorm2d(256)
        self.cv6 = nn.Conv2d(256, 512, 2, stride=2, bias=False)
        self.bn_cv6 = nn.BatchNorm2d(512)

        ## Latent representation of mean and std
        self.fc_mean = nn.Linear(512*CONV_OUT_WIDTH*CONV_OUT_HEIGHT, self.stl_latent_size)
        self.fc_logvar = nn.Linear(512*CONV_OUT_WIDTH*CONV_OUT_HEIGHT, self.stl_latent_size)
        self.fc_latent = nn.Linear(self.stl_latent_size, 512*CONV_OUT_WIDTH*CONV_OUT_HEIGHT)

        # Decoder dropout layer
        # self.dropout = nn.Dropout(p=0.1)

        ## Decoder
        self.dcv1 = nn.ConvTranspose2d(512, 256, 3, stride=2, bias=False)
        self.bn_dcv1 = nn.BatchNorm2d(256)
        self.dcv2 = nn.ConvTranspose2d(256, 128, 5, stride=2, bias=False)
        self.bn_dcv2 = nn.BatchNorm2d(128)
        self.dcv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, bias=False)
        self.bn_dcv3 = nn.BatchNorm2d(64)
        self.dcv4 = nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False)
        self.bn_dcv4 = nn.BatchNorm2d(32)
        self.dcv5 = nn.ConvTranspose2d(32, 16, 4, stride=(1,2), bias=False)
        self.bn_dcv5 = nn.BatchNorm2d(16)
        self.dcv6 = nn.ConvTranspose2d(16, self.frame_stack_size, 5, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, nn.init.calculate_gain('linear'))
                m.bias.data.fill_(0)

    def encode(self, inputs):
        x = F.relu(self.bn_cv1(self.cv1(inputs)))
        # print("cv1 {}".format(x.size()))
        x = F.relu(self.bn_cv2(self.cv2(x)))
        # print("cv2 {}".format(x.size()))
        x = F.relu(self.bn_cv3(self.cv3(x)))
        # print("cv3 {}".format(x.size()))
        x = F.relu(self.bn_cv4(self.cv4(x)))
        # print("cv4 {}".format(x.size()))
        x = F.relu(self.bn_cv5(self.cv5(x)))
        # print("cv5 {}".format(x.size()))
        x = F.relu(self.bn_cv6(self.cv6(x)))
        # print("cv6 {}".format(x.size()))

        x = x.view(-1, 512*CONV_OUT_WIDTH*CONV_OUT_HEIGHT)

        return self.fc_mean(x), self.fc_logvar(x)

    def decode(self, z):
        # x = self.fc_latent(self.dropout(z))
        x = self.fc_latent(z)

        x = x.view(-1, 512, CONV_OUT_WIDTH, CONV_OUT_HEIGHT)

        x = F.relu(self.bn_dcv1(self.dcv1(x)))
        # print("dcv1 {}".format(x.size()))
        x = F.relu(self.bn_dcv2(self.dcv2(x)))
        # print("dcv2 {}".format(x.size()))
        x = F.relu(self.bn_dcv3(self.dcv3(x)))
        # print("dcv3 {}".format(x.size()))
        x = F.relu(self.bn_dcv4(self.dcv4(x)))
        # print("dcv4 {}".format(x.size()))
        x = F.relu(self.bn_dcv5(self.dcv5(x)))
        # print("dcv5 {}".format(x.size()))
        x = F.sigmoid(self.dcv6(x))
        # print("dcv6 {}".format(x.size()))

        padh = int((x.size(2) - OUT_HEIGHT) / 2)
        padw = int((x.size(3) - OUT_WIDTH) / 2)

        x = x[:,:,padh:OUT_HEIGHT+padh,padw:OUT_WIDTH+padw]
        # import pdb; pdb.set_trace()

        return x

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, x, deterministic=False, detach_latent=False):
        mean, logvar = self.encode(x)
        latent = self.reparameterize(mean, logvar)

        if deterministic:
            latent = mean

        if detach_latent:
            latent = latent.detach()

        encoded = self.decode(latent)

        return latent, encoded, mean, logvar

    def set_requires_grad(self, requires_grad=True):
        for p in self.parameters():
            p.requires_grad = requires_grad

class STL(nn.Module):
    def __init__(self, config):
        super(STL, self).__init__()
        self.device = torch.device(config.get('device'))
        self.stl_latent_size = config.get('stl_latent_size')
        self.stl_hidden_size = config.get('stl_hidden_size')
        self.stl_layer_count = config.get('stl_layer_count')
        self.frame_stack_size = config.get('frame_stack_size')

        layers = [
            nn.Linear(
                self.frame_stack_size * State.size(),
                self.stl_hidden_size,
                bias=False,
            ),
            nn.BatchNorm1d(self.stl_hidden_size),
            nn.ReLU(True),
        ]

        for i in range(self.stl_layer_count):
            layers += [
                nn.Linear(
                    self.stl_hidden_size,
                    self.stl_hidden_size,
                    bias=False,
                ),
                nn.BatchNorm1d(self.stl_hidden_size),
                nn.ReLU(True),
            ]

        layers += [
            nn.Linear(
                self.stl_hidden_size,
                self.stl_latent_size,
            ),
        ]

        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data, nn.init.calculate_gain('linear'))
                    m.bias.data.fill_(0)
                else:
                    nn.init.xavier_normal_(m.weight.data, nn.init.calculate_gain('relu'))

    def forward(self, state):
        return self.layers(state)

