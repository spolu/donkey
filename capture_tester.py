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
from simulation import ANGLES_WINDOW

# import pdb; pdb.set_trace()

class Tester:
    def __init__(self, args):
        self.config = Config('configs/capture_trainer.json')

        if args.cuda != None:
            self.config.override('cuda', args.cuda)

        torch.manual_seed(self.config.get('seed'))
        random.seed(self.config.get('seed'))
        if self.config.get('cuda'):
            torch.cuda.manual_seed(self.config.get('seed'))

        self.cuda = self.config.get('cuda')
        self.hidden_size = self.config.get('hidden_size')

        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        self.capture = Capture(args.capture_dir, self.device)
        self.model = ResNet(self.config, ANGLES_WINDOW+1).to(self.device)

        self.load_dir = args.load_dir

        self.loss = nn.MSELoss()

        if self.cuda:
            self.model.load_state_dict(
                torch.load(self.load_dir + "/model.pt"),
            )
        else:
            self.model.load_state_dict(
                torch.load(self.load_dir + "/model.pt", map_location='cpu'),
                # torch.load(self.load_dir + "/model.pt", map_location=lambda storage, loc: storage), #some how it only work this way for me
            )

        self.test_loader = torch.utils.data.DataLoader(
            self.capture,
            batch_size=1,
            shuffle=True,
            num_workers=0,
        )

    def test(self):
        self.model.eval()
        loss_meter = Meter()

        for i, (cameras, values) in enumerate(self.test_loader):
            outputs = self.model(cameras)

            loss = self.loss(outputs, values)

            print(
                ("LOSS {:.4f}\n" + \
                 "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}  {:.2f}\n" + \
                 "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}  {:.2f}").format(
                    loss.item(),
                    values[0][0].item(),
                    values[0][1].item(),
                    values[0][2].item(),
                    values[0][3].item(),
                    values[0][4].item(),
                    values[0][5].item(),
                    outputs[0][0].item(),
                    outputs[0][1].item(),
                    outputs[0][2].item(),
                    outputs[0][3].item(),
                    outputs[0][4].item(),
                    outputs[0][5].item(),
                )
            )

            loss_meter.update(loss.item())

        print(
            ("TEST " + \
             "avg/min/max L {:.4f} {:.4f} {:.4f}").
            format(
                loss_meter.avg,
                loss_meter.min,
                loss_meter.max,
            )
        )


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--load_dir', type=str, help="path to saved models directory")
    parser.add_argument('--capture_dir', type=str, help="path to saved captured data")

    parser.add_argument('--cuda', type=str2bool, help="config override")

    args = parser.parse_args()

    tester = Tester(args)
    tester.test()
