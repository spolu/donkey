import math
import sys
import random
import argparse
import signal
import os.path

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

from utils import Config, str2bool, Meter
from capture import Capture, CaptureSet
from capture.models import ConvNet

# import pdb; pdb.set_trace()

class Trainer:
    def __init__(self, args):
        self.config = Config('configs/capture_trainer_recurring.json')

        if args.cuda != None:
            self.config.override('cuda', args.cuda)

        torch.manual_seed(self.config.get('seed'))
        random.seed(self.config.get('seed'))
        if self.config.get('cuda'):
            torch.cuda.manual_seed(self.config.get('seed'))

        self.cuda = self.config.get('cuda')
        self.learning_rate = self.config.get('learning_rate')
        self.hidden_size = self.config.get('hidden_size')
        self.sequence_size = self.config.get('sequence_size')

        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        if not args.train_capture_set_dir:
            raise Exception("Required argument: --train_capture_set_dir")
        self.train_capture_set = CaptureSet(args.train_capture_set_dir, self.device)
        if not args.test_capture_set_dir:
            raise Exception("Required argument: --test_capture_set_dir")
        self.test_capture_set = CaptureSet(args.test_capture_set_dir, self.device)

        self.model = ConvNet(self.config, 3, 3).to(self.device)

        self.save_dir = args.save_dir
        self.load_dir = args.load_dir

        self.optimizer = optim.Adam(
            self.model.parameters(),
            self.learning_rate,
        )
        self.mse_loss = nn.MSELoss()

        if self.load_dir:
            if self.cuda:
                self.model.load_state_dict(
                    torch.load(self.load_dir + "/model.pt"),
                )
                self.optimizer.load_state_dict(
                    torch.load(self.load_dir + "/optimizer.pt"),
                )
            else:
                self.model.load_state_dict(
                    torch.load(self.load_dir + "/model.pt", map_location='cpu'),
                )
                self.optimizer.load_state_dict(
                    torch.load(self.load_dir + "/optimizer.pt", map_location='cpu'),
                )

        self.batch_count = 0

    def loss(self, capture):
        pass

    def batch_train(self):
        self.model.train()
        loss_meter = Meter()

        for i in range(self.train_capture_set.size()):
            capture = self.train_capture_set.get_capture(i)

            progresses = []
            positions = []
            angles = []

            hidden = torch.zeros(1, self.hidden_size).to(self.device)

            for j in range(len(capture.ready)/self.sequence_size):
                # Detach hidden and results (used in computation of loss for continuity)
                hidden = hidden.detach()
                if j > 0:
                    progresses[self.sequence_size*j-1] = progresses[self.sequence_size*j-1].detach()
                    positions[self.sequence_size*j-1] = positions[self.sequence_size*j-1].detach()
                    angles[self.sequence_size*j-1] = angles[self.sequence_size*j-1].detach()

                for k in range(self.sequence_size):
                    outputs = self.model(cameras)


