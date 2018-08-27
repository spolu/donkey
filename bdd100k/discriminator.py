import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import pdb; pdb.set_trace()

class PatchGAN(nn.Module):
    def __init__(self, config, input_channel_count):
        super(PatchGAN, self).__init__()
        self.device = torch.device(config.get('device'))

        self.gan_first_conv_filters_count = config.get('gan_first_conv_filters_count')
        self.gan_layers_count = config.get('gan_layers_count')

        nf = self.gan_first_conv_filter

        sequences = [[
            nn.Conv2d(
                input_channel_count,
                nf,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.LeakyReLU(0.2, True),
        ]]

        # Layers Stride 2
        for i in range(1, self.gen_layers_count):
            mult = 2 ** i
            sequences += [[
                nn.Conv2d(
                    min(nf * mult, 512),
                    min(nf * mult * 2, 512),
                    kernel_size=4, stride=2, padding=1,
                ),
                nn.InstanceNorm2d(nf * mult * 2),
                nn.LeakyReLU(0.2, True),
            ]]

        # Layer Stride 1
        sequences += [[
            nn.Conv2d(
                min(nf * (2 ** (self.gen_layers_count - 1)), 512),
                min(nf * (2 ** self.gen_layers_count), 512),
                kernel_size=4, stride=1, padding=1,
            ),
            nn.InstanceNorm2d(nf * mult * 2),
            nn.LeakyReLU(0.2, True),
        ]]

        # Filter 1 layer.
        sequences += [[
            nn.Conv2d(
                min(nf * (2 ** self.gen_layers_count), 512),
                1,
                kernel_size=4, stride=1, padding=1),
        ]]

        for n in range(len(sequences)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def forward(self, x):
        res = [x]
        for n in range(self.gen_layers_count + 2):
            model = getattr(self, 'model'+str(n))
            res.append(model(res[-1]))
        return res[1:]
