import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import pdb; pdb.set_trace()

class Encoder(nn.Module):
    def __init__(self, config, input_channel_count):
        self.device = torch.device(config.get('device'))

        self.enc_downsampling_count = config.get('enc_downsampling_count')
        self.enc_feature_count = config.get('enc_feature_count')
        self.enc_first_conv_filter_count = config.get('enc_first_conv_filter_count')

        nf = self.enc_first_conv_filter_count

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channel_count, nf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
        ]

        # Downsampling.
        for i in range(self.enc_downsampling_count):
            mult = 2 ** i
            layers += [
                nn.Conv2d(nf * mult, nf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(nf * mult * 2),
                nn.ReLU(True),
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
            nn.Conv2d(nf, self.enc_feature_count, kernel_size=7, padding=0),
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

    def forward(self, x, instance_map):
        out = self.layers(x)

        # Instance-wise average pooling.
        pool = out.clone()
        instance_list = np.unique(instane_map.cpu().numpy().astype(int))
        for i in instance_list:
            for b in range(x.size()[0]):
                indices = (instance_map[b:b+1] == int(i)).nonzero()
                for j in range(self.enc_feature_count):
                    ins = out[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]
                    mean = torch.mean(ins).expand_as(ins)
                    pool[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean

        return pool

    def feature_count():
        return self.enc_feature_count
