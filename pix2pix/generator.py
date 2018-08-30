import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import pdb; pdb.set_trace()

class ResidualBlock(nn.Module):
    def __init__(self, config, dim):
        super(ResidualBlock, self).__init__()
        self.gen_residual_dropout = config.get('gen_residual_dropout')

        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
        ]

        if self.gen_residual_dropout > 0.0:
            layers += [
                nn.Dropout(self.gen_residual_dropout)
            ]

        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return (x + self.layers(x))


class Generator(nn.Module):
    def __init__(self, config, input_channel_count, output_channel_count):
        super(Generator, self).__init__()
        self.device = torch.device(config.get('device'))

        self.gen_downsampling_count = config.get('gen_downsampling_count')
        self.gen_residual_block_count = config.get('gen_residual_block_count')
        self.gen_first_conv_filter_count = config.get('gen_first_conv_filter_count')

        nf = self.gen_first_conv_filter_count

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channel_count, nf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
        ]

        # Downsampling.
        for i in range(self.gen_downsampling_count):
            mult = 2 ** i
            layers += [
                nn.Conv2d(nf * mult, nf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(nf * mult * 2),
                nn.ReLU(True),
            ]

        # Residual Blocks:
        mult = 2**self.gen_downsampling_count
        for i in range(self.gen_residual_block_count):
            layers += [
                ResidualBlock(config, nf * mult),
            ]

        # Upsampling.
        for i in range(self.gen_downsampling_count):
            mult = 2 ** (self.gen_downsampling_count - i)
            layers += [
                nn.ConvTranspose2d(nf * mult, int(nf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(nf * mult / 2)),
                nn.ReLU(True),
            ]

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf, output_channel_count, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x):
        return self.layers(x)
