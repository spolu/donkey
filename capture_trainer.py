import sys
import random
import argparse
import signal
import os.path

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from utils import Config, str2bool, Meter
from capture import Capture
from capture.models import ResNet

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
        self.grad_norm_max = self.config.get('grad_norm_max')

        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        if not args.capture_dir:
            raise Exception("Required argument: --capture_dir")
        self.capture = Capture(args.capture_dir, self.device)
        self.model = ResNet(self.config, 2).to(self.device)

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
                    # torch.load(self.load_dir + "/model.pt", map_location=lambda storage, loc: storage), #some how it only work this way for me
                )
                self.optimizer.load_state_dict(
                    torch.load(self.load_dir + "/optimizer.pt", map_location='cpu'),
                    # torch.load(self.load_dir + "/optimizer.pt", map_location=lambda storage, loc: storage), #some how it only work this way for me
                )

        self.batch_count = 0

        self.train_loader = torch.utils.data.DataLoader(
            self.capture,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def batch_train(self):
        loss_meter = Meter()

        for i, (cameras, values) in enumerate(self.train_loader):
            outputs = self.model(cameras)

            loss = self.loss(outputs, values)
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(
            ("EPISODE {} " + \
             "avg/min/max L {:.4f} {:.4f} {:.4f}").
            format(
                self.episode,
                loss_meter.avg,
                loss_meter.min,
                loss_meter.max,
            )
        )

    def train(self):
        self.episode = 0
        self.model.train()

        while True:
            self.batch_train()
            sys.stdout.flush()
            self.episode += 1

            if self.episode % 20 == 0 and self.save_dir:
                print("Saving models and optimizer: save_dir={}".format(self.save_dir))
                torch.save(self.model.state_dict(), self.save_dir + "/model.pt")
                torch.save(self.optimizer.state_dict(), self.save_dir + "/optimizer.pt")

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--save_dir', type=str, help="directory to save models")
    parser.add_argument('--load_dir', type=str, help="path to saved models directory")
    parser.add_argument('--capture_dir', type=str, help="path to saved captured data")

    parser.add_argument('--cuda', type=str2bool, help="config override")

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
