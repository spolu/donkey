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

# import pdb; pdb.set_trace()

class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.cv1_rs = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1_rs = nn.BatchNorm2d(planes)
        self.cv2_rs = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2_rs = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.cv1_rs(x)
        out = self.bn1_rs(out)
        out = F.relu(out, inplace=True)

        out = self.cv2_rs(out)
        out = self.bn2_rs(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        return out

class HeadBlock(nn.Module):
    def __init__(self, hidden_size, nonlinearity, output_size, stride=1):
        super(HeadBlock, self).__init__()
        self.fc1_hd = nn.Linear(hidden_size, hidden_size)
        self.fc2_hd = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal_(self.fc1_hd.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc2_hd.weight.data, nn.init.calculate_gain(nonlinearity))
        self.fc1_hd.bias.data.fill_(0)
        self.fc2_hd.bias.data.fill_(0)

    def forward(self, x):
        out = self.fc1_hd(x)
        out = F.relu(out, inplace=True)
        out = self.fc2_hd(out)

        return out


class ResNet(nn.Module):
    def __init__(self, config, stack_count, value_count):
        super(ResNet, self).__init__()
        self.hidden_size = config.get('hidden_size')
        self.config = config
        self.inplanes = 64

        self.stack_count = stack_count
        self.value_count = value_count

        self.cv1 = nn.Conv2d(
            self.stack_count, 64,
            kernel_size=7, stride=2, padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rs11 = self._make_residual_layer(64, 1, stride=1)
        self.rs12 = self._make_residual_layer(64, 1, stride=2)
        self.rs21 = self._make_residual_layer(128, 1, stride=1)
        self.rs22 = self._make_residual_layer(128, 1, stride=2)

        self.avgp = nn.AvgPool2d(8, stride=1)

        self.fc1 = nn.Linear(3*128, self.hidden_size)

        # Value head.
        self.hd_v = HeadBlock(self.hidden_size, 'linear', self.value_count)

        nn.init.xavier_normal_(self.fc1.weight.data, nn.init.calculate_gain('relu'))
        self.fc1.bias.data.fill_(0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.train()

    def _make_residual_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes, kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.cv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxp(x)

        x = self.rs11(x)
        x = self.rs12(x)
        x = self.rs21(x)
        x = self.rs22(x)
        # x = self.rs3(x)

        x = self.avgp(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)

        return self.hd_v(x)

class Trainer:
    def __init__(self, args):
        self.config = Config('configs/capture_trainer.json')

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
            outputs = self.model(cameras)

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
            outputs = self.model(cameras)

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
