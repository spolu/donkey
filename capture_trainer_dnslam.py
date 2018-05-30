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
from capture import Capture, CaptureSet
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
        self.hidden_size = self.config.get('hidden_size')
        self.sequence_size = self.config.get('sequence_size')

        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        if not args.capture_set_train_dir:
            raise Exception("Required argument: --capture_set_train_dir")
        self.train_capture_set = CaptureSet(args.capture_set_train_dir, self.device)
        if not args.capture_set_test_dir:
            raise Exception("Required argument: --capture_set_test_dir")
        self.test_capture_set = CaptureSet(args.capture_set_test_dir, self.device)

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

    def loss(self, capture, r, progresses, positions, angles, speeds):
        PROGRESS_DELTA_CAP = 0.01
        POSITION_DELTA_CAP = 0.01
        ANGLE_DELTA_CAP = 0.4
        SPEED_DELTA_CAP = 0.1

        loss = 0

        for i in r:
            # reference positions
            annotated_loss = 0
            item = capture.get_item(i)
            if 'annotated_track_position' in item:
                annotated_loss += F.mse_loss(
                    positions[i],
                    torch.zeros(1).to(self.device) + item['annotated_track_position'],
                )
            if 'annotated_progress' in item:
                annotated_loss += F.mse_loss(
                    progresses[i],
                    torch.zeros(1).to(self.device) + item['annotated_progress'],
                )

            if i == 0:
                loss += annotated_loss
                continue

            # speed, angle integration
            delta_time = capture.get_item(i)['time'] - capture.get_item(i-1)['time']

            p = progresses[i-1] + delta_time * speeds[i-1] * torch.cos(angles[i-1])
            integration_loss = F.mse_loss(p, progresses[i])
            p = positions[i-1] + delta_time * speeds[i-1] * torch.sin(angles[i-1])
            integration_loss += F.mse_loss(p, positions[i])

            # delta progress cap
            # delta = progresses[i]-progresses[i-1]
            # progress_cap_loss = F.relu(torch.abs(delta) - PROGRESS_DELTA_CAP).squeeze(0)

            # delta position cap
            # delta = positions[i]-positions[i-1]
            # position_cap_loss = F.relu(torch.abs(delta) - POSITION_DELTA_CAP).squeeze(0)

            # delta angle cap
            # delta = angles[i]-angles[i-1]
            # angle_cap_loss = F.relu(torch.abs(delta) - ANGLE_DELTA_CAP).squeeze(0)

            # delta speed cap
            # delta = speeds[i]-speeds[i-1]
            # speed_cap_loss = F.relu(torch.abs(delta) - SPEED_DELTA_CAP).squeeze(0)

            # print(("LOSS " + \
            #        "reference {:.6f} " + \
            #        "integration {:.6f} " + \
            #        "progres_cap {:.6f} " + \
            #        "position_cap {:.6f} " + \
            #        "angle_cap {:.6f} " + \
            #        "speed_cap {:.6f}").format(
            #            annotated_loss,
            #            integration_loss,
            #            progress_cap_loss,
            #            position_cap_loss,
            #            angle_cap_loss,
            #            speed_cap_loss,
            #        ))

            # loss += (annotated_loss +
            #          integration_loss +
            #          progress_cap_loss +
            #          position_cap_loss +
            #          angle_cap_loss +
            #          speed_cap_loss)
            loss += annotated_loss + integration_loss

        return loss


    def batch_train(self):
        self.model.train()
        loss_meter = Meter()

        for i in range(self.train_capture_set.size()):
            capture = self.train_capture_set.get_capture(i)
            hidden = torch.zeros(1, self.hidden_size).to(self.device)
            sequence = capture.sequence()

            progresses = []
            positions = []
            angles = []
            speeds = []

            for j in range(int(sequence.size(0)/self.sequence_size)):
                hidden = hidden.detach()
                if j > 0:
                    progresses[self.sequence_size*j-1] = progresses[self.sequence_size*j-1].detach()
                    positions[self.sequence_size*j-1] = positions[self.sequence_size*j-1].detach()
                    angles[self.sequence_size*j-1] = angles[self.sequence_size*j-1].detach()
                    speeds[self.sequence_size*j-1] = speeds[self.sequence_size*j-1].detach()

                for k in range(self.sequence_size):
                    progress, position, angle, speed, hidden = self.model(
                        sequence[self.sequence_size*j+k].unsqueeze(0), hidden,
                    )

                    progresses.append(progress[0])
                    positions.append(position[0])
                    angles.append(angle[0])
                    speeds.append(speed[0])

                loss = self.loss(
                    capture,
                    range(self.sequence_size*j, self.sequence_size*(j+1)),
                    progresses, positions, angles, speeds,
                )
                loss_meter.update(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.episode % 5 == 0:
                # Saving capture.
                print("Saving capture...")
                for j in range(capture.__len__()):
                    if j < len(progresses):
                        capture.update_item(j, {
                            'integrated_progress': progresses[j].item(),
                            'integrated_track_position': positions[j].item(),
                            'integrated_track_angle': angles[j].item(),
                        }, save=True)

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

        with torch.set_grad_enabled(False):

            for i in range(self.test_capture_set.size()):
                capture = self.test_capture_set.get_capture(i)
                hidden = torch.zeros(1, self.hidden_size).to(self.device)
                sequence = capture.sequence()

                progresses = []
                positions = []
                angles = []
                speeds = []

                for j in range(int(sequence.size(0)/self.sequence_size)):
                    for k in range(self.sequence_size):
                        progress, position, angle, speed, hidden = self.model(
                            sequence[self.sequence_size*j+k].unsqueeze(0), hidden,
                        )

                        progresses.append(progress[0])
                        positions.append(position[0])
                        angles.append(angle[0])
                        speeds.append(speed[0])

                    loss = self.loss(
                        capture,
                        range(self.sequence_size*j, self.sequence_size*(j+1)),
                        progresses, positions, angles, speeds,
                    )
                    loss_meter.update(loss.item())

                # Saving capture.
                print("Saving capture...")
                for j in range(capture.__len__()):
                    if j < len(progresses):
                        capture.update_item(j, {
                            'integrated_progress': progresses[j].item(),
                            'integrated_track_position': positions[j].item(),
                            'integrated_track_angle': angles[j].item(),
                        }, save=True)

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
    parser.add_argument('--capture_set_train_dir', type=str, help="path to train captured data set")
    parser.add_argument('--capture_set_test_dir', type=str, help="path to test captured data set")

    parser.add_argument('--cuda', type=str2bool, help="config override")

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
