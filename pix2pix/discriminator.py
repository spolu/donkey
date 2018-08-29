import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import pdb; pdb.set_trace()

class PatchGAN(nn.Module):
    def __init__(self, config, input_channel_count):
        super(PatchGAN, self).__init__()
        self.device = torch.device(config.get('device'))

        self.gan_first_conv_filter_count = config.get('gan_first_conv_filter_count')
        self.gan_layer_count = config.get('gan_layer_count')

        nf = self.gan_first_conv_filter_count

        layer_groups = [[
            nn.Conv2d(
                input_channel_count,
                nf,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.LeakyReLU(0.2, True),
        ]]

        # Layers Stride 2
        for i in range(1, self.gan_layer_count):
            mult = 2 ** i
            layer_groups += [[
                nn.Conv2d(
                    min(nf * mult, 512),
                    min(nf * mult * 2, 512),
                    kernel_size=4, stride=2, padding=1,
                ),
                nn.InstanceNorm2d(nf * mult * 2),
                nn.LeakyReLU(0.2, True),
            ]]

        # Layer Stride 1
        layer_groups += [[
            nn.Conv2d(
                min(nf * (2 ** (self.gan_layer_count - 1)), 512),
                min(nf * (2 ** self.gan_layer_count), 512),
                kernel_size=4, stride=1, padding=1,
            ),
            nn.InstanceNorm2d(nf * mult * 2),
            nn.LeakyReLU(0.2, True),
        ]]

        # Filter 1 layer.
        layer_groups += [[
            nn.Conv2d(
                min(nf * (2 ** self.gan_layer_count), 512),
                1,
                kernel_size=4, stride=1, padding=1),
        ]]

        for n in range(len(layer_groups)):
            setattr(
                self,
                'layer'+str(n),
                nn.Sequential(*layer_groups[n]),
            )

    def forward(self, x):
        res = [x]
        for n in range(self.gan_layer_count + 2):
            layer = getattr(self, 'layer'+str(n))
            res.append(layer(res[-1]))
        return res[1:]

class Discriminator(nn.Module):
    def __init__(self, config, input_channel_count):
        super(Discriminator, self).__init__()
        self.device = torch.device(config.get('device'))

        self.gan_scale_count = config.get('gan_scale_count')
        self.gan_layer_count = config.get('gan_layer_count')

        for i in range(self.gan_scale_count):
            d = PatchGAN(config, input_channel_count)
            for n in range(self.gan_layer_count + 2):
                setattr(
                    self,
                    'scale'+str(i)+'_layer'+str(j),
                    getattr(d, 'layer'+str(j)),
                )

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False,
        )

    def scale_forward(self, layers, x):
        res = [x]
        for i in range(len(layers)):
            res.append(layers[i](res[-1]))
        return res[1:]

    def forward(self, x):
        res = []
        downsampled = x

        for i in range(self.gan_scale_count):
            layers = [
                getattr(
                    self,
                    'scale'+str(self.gan_scale_count-1-i)+'_layer'+str(j)
                ) for j in range(self.gan_layer_count + 2)
            ]
            res.append(self.scale_forward(layers, downsampled))

            if i != (self.gan_scale_count-1):
                downsampled = self.downsample(downsampled)

        return res
