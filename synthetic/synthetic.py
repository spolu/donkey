import cv2
import sys

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from synthetic import Decoder, State
from capture import CaptureSet
from utils import Meter

# import pdb; pdb.set_trace()

class Synthetic:
    def __init__(self, config, save_dir=None, load_dir=None):
        self.config = config
        self.device = torch.device(config.get('device'))

        self.decoder = Decoder(config).to(self.device)

        self.save_dir = save_dir
        self.load_dir = load_dir

        if self.load_dir:
            if config.get('device') != 'cpu':
                self.policy.load_state_dict(
                    torch.load(self.load_dir + "/decoder.pt"),
                )
            else:
                self.policy.load_state_dict(
                    torch.load(self.load_dir + "/decoder.pt", map_location='cpu'),
                )

    def _decoder_capture_loader(self, item):
        state = torch.from_numpy(State(
            item['simulation_track_randomization'],
            item['simulation_position'],
            item['simulation_velocity'],
            item['simulation_angular_velocity'],
            item['simulation_track_coordinates'],
            item['simulation_track_angle'],
        ).vector()).float().to(self.device)

        camera = torch.from_numpy(
            cv2.imdecode(
                np.fromstring(item['camera'], np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )[50:] / 255.0
        ).float().to(self.device)

        return state, camera

    def decoder_initialize_training(
            self,
            train_capture_set_dir,
            test_capture_set_dir,
    ):
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(),
            self.config.get('learning_rate'),
        )

        if self.load_dir:
            if self.config.get('device') != 'cpu':
                self.decoder_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/decoder_optimizer.pt"),
                )
            else:
                self.decoder_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/decoder_optimizer.pt", map_location='cpu'),
                )

        self.train_capture_set = CaptureSet(
            train_capture_set_dir,
            loader=self._decoder_capture_loader
        )
        self.test_capture_set = CaptureSet(
            test_capture_set_dir,
            loader=self._decoder_capture_loader
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_capture_set,
            batch_size=self.config.get('batch_size'),
            shuffle=True,
            num_workers=4,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_capture_set,
            batch_size=self.config.get('batch_size'),
            shuffle=False,
            num_workers=0,
        )

        self.batch_count = 0
        self.best_test_loss = 9999.0

    def decoder_batch_train(self):
        self.decoder.train()
        loss_meter = Meter()

        for i, (states, cameras) in enumerate(self.train_loader):
            reconstructs, means, logvars = self.decoder(
                states.detach(),
            )

            self.decoder_optimizer.zero_grad()

            bce_loss = F.binary_cross_entropy(
                reconstructs, cameras,
            )
            mse_loss = F.mse_loss(
                reconstructs, cameras,
            )
            kld_loss = -0.5 * torch.sum(
                1 + logvars - means.pow(2) - logvars.exp()
            )
            kld_loss /= states.size(0)

            bce_loss.backward()
            loss_meter.update(bce_loss.item())

            self.decoder_optimizer.step()

            # cv2.imwrite('vae_observation.jpg', (255 * observations_batch[0][0].to('cpu')).detach().numpy())
            # cv2.imwrite('vae_reconstruct.jpg', (255 * reconstructs[0][0].to('cpu')).detach().numpy())

            print(
                ("TRAIN {} batch {} " + \
                 "bce_loss {:.5f} " + \
                 "mse_loss {:.5f} " + \
                 "kld_loss {:.5f}").
                format(
                    self.batch_count,
                    i,
                    bce_loss.item(),
                    mse_loss.item(),
                    kld_loss.item(),
                ))
            sys.stdout.flush()

        self.batch_count += 1
        return loss_meter

    def decoder_batch_test(self):
        self.decoder.eval()
        loss_meter = Meter()

        for i, (states, cameras) in enumerate(self.test_loader):
            reconstructs, means, logvars = self.decoder(
                states.detach(), deterministic=True,
            )

            bce_loss = F.binary_cross_entropy(
                reconstructs, cameras,
            )
            mse_loss = F.mse_loss(
                reconstructs, cameras,
            )
            kld_loss = -0.5 * torch.sum(
                1 + logvars - means.pow(2) - logvars.exp()
            )
            kld_loss /= states.size(0)

            loss_meter.update(bce_loss.item())

            print(
                ("TEST {} batch {}" + \
                 "bce_loss {:.5f} " + \
                 "mse_loss {:.5f} " + \
                 "kld_loss {:.5f}").
                format(
                    self.batch_count,
                    i,
                    bce_loss.item(),
                    mse_loss.item(),
                    kld_loss.item(),
                ))
            sys.stdout.flush()

            # TODO store policy if we beated the test

        return loss_meter
