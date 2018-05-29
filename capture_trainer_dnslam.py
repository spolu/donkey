import sys
import random
import argparse
import signal
import os.path

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import Config, str2bool, Meter
from capture import Capture, CaptureSequenceSet
from capture.models import DNSLAM

# import pdb; pdb.set_trace()

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

        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        if not args.capture_set_train_dir:
            raise Exception("Required argument: --capture_set_train_dir")
        self.train_capture_set = CaptureSequenceSet(args.capture_set_train_dir, self.device)
        if not args.capture_set_test_dir:
            raise Exception("Required argument: --capture_set_test_dir")
        self.test_capture_set = CaptureSequenceSet(args.capture_set_test_dir, self.device)

        self.model = DNSLAM(self.config, 3).to(self.device)

        self.save_dir = args.save_dir
        self.load_dir = args.load_dir

        self.optimizer = optim.Adam(
            self.model.parameters(),
            self.learning_rate,
        )

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
                    # torch.load(self.load_dir + "/model.pt", map_location=lambda storage, loc: storage), #some how it only work this way for me
                )
                self.optimizer.load_state_dict(
                    torch.load(self.load_dir + "/optimizer.pt", map_location='cpu'),
                    # torch.load(self.load_dir + "/optimizer.pt", map_location=lambda storage, loc: storage), #some how it only work this way for me
                )

        self.batch_count = 0

        self.train_loader = torch.utils.data.DataLoader(
            self.train_capture_set,
            batch_size=1,
            shuffle=True,
            num_workers=0,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_capture_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    def loss(self, capture, progresses, positions, angles, speeds):
        loss = 0

        SPEED_DELTA_CAP = 0.3
        ANGLE_DELTA_CAP = 0.4

        for i in range(1, capture.size()):
            # reference positions
            item = capture.get_item(i)
            if item['reference_track_position']:
                loss += F.mse_loss(positions[i], torch.zeros(0) + item['reference_track_positions'])
            if item['reference_progress']:
                loss += F.mse_loss(progress[i], torch.zeros(0) + item['reference_progress'])

            # on-track
            loss += F.relu(torch.abs(positions[i]) - 1.0)

            if i == 0:
                continue

            # speed derivation
            delta_time = capture.get_item(i)['time'] - capture.get_item(i-1)['time']

            p = progresses[i-1] + delta_time * speeds[i-1] * torch.cos(angles[i-1])
            loss += F.mse_loss(p, progresses[i])

            p = positions[i-1] + detla_time * speeds[i-1] * torch.sin(angles[i-1])
            loss += F.mse_loss(p, positions[i])

            # delta speed cap
            delta = speeds[i]-speed[i-1]
            loss += F.relu(torch.abs(delta) - SPEED_DELTA_CAP)

            # delta angle cap
            delta = angles[i]-angles[i-1]
            loss += F.relu(torch.abs(delta) - ANGLE_DELTA_CAP)



    def batch_train(self):
        self.model.train()
        loss_meter = Meter()

        for i, capture in enumerate(self.train_loader):
            hidden = torch.zeros(1, self.hidden_size)
            sequence = capture.sequence().transpose(0, 1)

            progresses = []
            positions = []
            angles = []
            speeds = []

            for i in range(sequence.size(0)):
                progress, position, angle, speed, hidden = self.model(sequence[i], hidden)

                progresses.append(progress[0])
                positions.append(position[0])
                angles.append(angle[0])
                speeds.append(speed[0])

            loss = self.loss(capture, progresses, positions, angles, speeds)
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

        for i, capture in enumerate(self.train_loader):
            hidden = torch.zeros(1, self.hidden_size)
            sequence = sequence.transpose(0, 1)

            progresses = []
            positions = []
            angles = []
            speeds = []

            for i in range(sequence.size(0)):
                progress, position, angle, speed, hidden = self.model(sequence[i], hidden)

                progresses.append(progress[0])
                positions.append(position[0])
                angles.append(angle[0])
                speeds.append(speed[0])

            loss = self.loss(capture, progresses, positions, angles, speeds)
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

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--save_dir', type=str, help="directory to save models")
    parser.add_argument('--load_dir', type=str, help="path to saved models directory")
    parser.add_argument('--capture_set_train_dir', type=str, help="path to train captured data set")
    parser.add_argument('--capture_set_test_dir', type=str, help="path to test captured data set")

    parser.add_argument('--cuda', type=str2bool, help="config override")

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
