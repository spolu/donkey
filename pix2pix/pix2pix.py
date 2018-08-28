import cv2
import sys
import os

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from bdd100k import Generator, MultiscalePatchGAN, Encoder

# import pdb; pdb.set_trace()

class BDD100k:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device'))

        self.generator = Generator(config, ).to(self.device)
        self.stl = STL(config).to(self.device)

        self.save_dir = config.get('bdd100k_save_dir')
        self.load_dir = config.get('bdd100k_load_dir')


