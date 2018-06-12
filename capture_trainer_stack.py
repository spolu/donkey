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
from capture import Capture, CaptureSet, StackCaptureSet
from capture.models import ResNet

# import pdb; pdb.set_trace()

class Trainer:
    def __init__(self, args):
        self.config = Config('configs/capture_trainer_stack.json')

        if args.cuda != None:
            self.config.override('cuda', args.cuda)

        torch.manual_seed(self.config.get('seed'))
        random.seed(self.config.get('seed'))
        if self.config.get('cuda'):
            torch.cuda.manual_seed(self.config.get('seed'))

        self.cuda = self.config.get('cuda')
        self.learning_rate = self.config.get('learning_rate')
        self.batch_size = self.config.get('batch_size')
        self.hidden_size = self.config.get('hidden_size')
        self.stack_size = self.config.get('stack_size')

        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        if not args.train_capture_set_dir:
            raise Exception("Required argument: --train_capture_set_dir")
        self.train_capture_set = StackCaptureSet(
            args.train_capture_set_dir, self.stack_size, self.device,
        )
        if not args.test_capture_set_dir:
            raise Exception("Required argument: --test_capture_set_dir")
        self.test_capture_set = StackCaptureSet(
            args.test_capture_set_dir, self.stack_size, self.device,
        )

        self.model = ResNet(self.config, 3 * self.stack_size, 3).to(self.device)

        self.save_dir = args.save_dir
        self.load_dir = args.load_dir

        self.optimizer = optim.Adam(
            self.model.parameters(),
            self.learning_rate,
        )
        self.loss = nn.MSELoss()

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

        self.train_loader = torch.utils.data.DataLoader(
            self.train_capture_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_capture_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    def batch_train(self):
        self.model.train()
        loss_meter = Meter()

        for i, (cameras, values) in enumerate(self.train_loader):
            outputs = self.model(cameras, torch.zeros(cameras.size(0), 1))
            loss = self.loss(outputs, values)
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(
            ("EPISODE {} avg/min/max L {:.6f} {:.6f} {:.6f}").
            format(
                self.episode,
                loss_meter.avg,
                loss_meter.min,
                loss_meter.max,
            )
        )
        return loss_meter.avg

    def batch_test(self):
        self.model.eval()
        loss_meter = Meter()

        for i, (cameras, values) in enumerate(self.test_loader):
            outputs = self.model(cameras, torch.zeros(cameras.size(0), 1))
            loss = self.loss(outputs, values)
            loss_meter.update(loss.item())

        print(
            ("TEST {} avg/min/max L {:.6f} {:.6f} {:.6f}").
            format(
                self.episode,
                loss_meter.avg,
                loss_meter.min,
                loss_meter.max,
            )
        )
        return loss_meter.avg

    def train(self):
        self.episode = 0
        self.best_test_loss = sys.float_info.max

        while True:
            self.batch_train()

            if self.episode % 10 == 0:
                loss = self.batch_test()
                if loss < self.best_test_loss:
                    self.best_test_loss = loss
                    if self.save_dir:
                        print(
                            "Saving models and optimizer: save_dir={}".
                            format(self.save_dir)
                        )
                        torch.save(
                            self.model.state_dict(),
                            self.save_dir + "/model.pt",
                        )
                        torch.save(
                            self.optimizer.state_dict(),
                            self.save_dir + "/optimizer.pt",
                        )

            self.episode += 1
            sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--save_dir', type=str, help="directory to save models")
    parser.add_argument('--load_dir', type=str, help="path to saved models directory")
    parser.add_argument('--train_capture_set_dir', type=str, help="path to train captured data")
    parser.add_argument('--test_capture_set_dir', type=str, help="path to test captured data")

    parser.add_argument('--cuda', type=str2bool, help="config override")

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
