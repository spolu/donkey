import cv2
import sys
import os

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from synthetic import Generator, Discriminator, State
from capture import CaptureSet
from utils import Meter
from reinforce import InputFilter

# import pdb; pdb.set_trace()

class Synthetic:
    def __init__(self, config, save_dir=None, load_dir=None):
        self.config = config
        self.device = torch.device(config.get('device'))
        self.input_filter = InputFilter(config)

        self.generator = Generator(config).to(self.device)

        self.save_dir = save_dir
        self.load_dir = load_dir

        if self.load_dir:
            if config.get('device') != 'cpu':
                self.generator.load_state_dict(
                    torch.load(self.load_dir + "/generator.pt"),
                )
            else:
                self.generator.load_state_dict(
                    torch.load(self.load_dir + "/generator.pt", map_location='cpu'),
                )

    def generate(self, state):
        generated, _, _ = self.generator(
            torch.from_numpy(
                state.vector()
            ).float().to(self.device).unsqueeze(0).detach(),
            deterministic=True,
        )
        return generated * 255.0

    def _generator_capture_loader(self, item):
        state = torch.from_numpy(State(
            item['simulation_track_randomization'],
            item['simulation_position'],
            item['simulation_velocity'],
            item['simulation_angular_velocity'],
            item['simulation_track_coordinates'],
            item['simulation_track_angle'],
        ).vector()).float().to(self.device)

        camera = torch.from_numpy(
            self.input_filter.apply(item['camera']) / 255.0
        ).float().to(self.device).unsqueeze(0)

        return state, camera

    def initialize_training(
            self,
            train_capture_set_dir,
            test_capture_set_dir,
    ):
        self.discriminator = Discriminator(self.config).to(self.device)

        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            self.config.get('learning_rate'),
        )
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            self.config.get('learning_rate'),
        )
        self.l1_loss_coeff = self.config.get('l1_loss_coeff')
        self.kld_loss_coeff = self.config.get('kld_loss_coeff')
        self.gan_loss_coeff = self.config.get('gan_loss_coeff')

        if self.load_dir:
            if self.config.get('device') != 'cpu':
                self.generator_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/generator_optimizer.pt"),
                )
                self.discriminator.load_state_dict(
                    torch.load(self.load_dir + "/discriminator.pt"),
                )
                self.discriminator_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/discriminator_optimizer.pt"),
                )
            else:
                self.generator_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/generator_optimizer.pt", map_location='cpu'),
                )
                self.discriminator.load_state_dict(
                    torch.load(self.load_dir + "/discriminator.pt", map_location='cpu'),
                )
                self.discriminator_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/discriminator_optimizer.pt", map_location='cpu'),
                )

        self.train_capture_set = CaptureSet(
            train_capture_set_dir,
            loader=self._generator_capture_loader
        )
        self.test_capture_set = CaptureSet(
            test_capture_set_dir,
            loader=self._generator_capture_loader
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_capture_set,
            batch_size=self.config.get('batch_size'),
            shuffle=True,
            num_workers=0,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_capture_set,
            batch_size=self.config.get('batch_size'),
            shuffle=False,
            num_workers=0,
        )

        self.batch_count = 0
        self.best_test_loss = 9999.0

    def _discriminator_loss(self, cameras, generated):
        pred_fake = self.discriminator(generated.detach())
        fake_loss = F.binary_cross_entropy(
            pred_fake,
            torch.zeros(*pred_fake.size()).to(self.device),
        )

        pred_real = self.discriminator(cameras.detach())
        real_loss = F.binary_cross_entropy(
            pred_real,
            torch.ones(*pred_real.size()).to(self.device),
        )

        return fake_loss, real_loss

    def _generator_loss(self, cameras, generated, means, logvars):
        # Do not detach as we want the gradients to flow from D to G.
        pred_gan = self.discriminator(generated)
        gan_loss = F.binary_cross_entropy(
            pred_gan,
            torch.ones(*pred_gan.size()).to(self.device),
        )

        l1_loss = F.l1_loss(
            generated, cameras,
        )

        kld_loss = -0.5 * torch.sum(
            1 + logvars - means.pow(2) - logvars.exp()
        )
        kld_loss /= generated.size(0) * generated.size(1) * generated.size(2)

        return l1_loss, kld_loss, gan_loss

    def batch_train(self):
        self.generator.train()
        self.discriminator.train()
        loss_meter = Meter()

        for i, (states, cameras) in enumerate(self.train_loader):
            generated, means, logvars = self.generator(
                states.detach(), deterministic=True,
            )

            # Discriminator pass.
            self.discriminator.set_requires_grad(True)
            self.discriminator_optimizer.zero_grad()

            fake_loss, real_loss = self._discriminator_loss(
                cameras, generated,
            )

            (fake_loss * 0.5 +
             real_loss * 0.5).backward()

            self.discriminator_optimizer.step()

            # Generator pass.
            self.discriminator.set_requires_grad(False)
            self.generator_optimizer.zero_grad()

            l1_loss, kld_loss, gan_loss = self._generator_loss(
                cameras, generated, means, logvars,
            )

            loss = (l1_loss * self.l1_loss_coeff +
                    # kld_loss * self.kld_loss_coeff +
                    gan_loss * self.gan_loss_coeff)
            loss.backward()

            self.generator_optimizer.step()

            loss_meter.update(loss.item())

            if i % 1000 == 0:
                if self.save_dir:
                    cv2.imwrite(
                        os.path.join(self.save_dir, '{}_synthetic_original.jpg'.format(i)),
                        (255 * cameras[0].squeeze(0).to('cpu')).detach().numpy(),
                    )
                    cv2.imwrite(
                        os.path.join(self.save_dir, '{}_synthetic_generated.jpg'.format(i)),
                        (255 * generated[0].squeeze(0).to('cpu')).detach().numpy(),
                    )

            print(
                ("TRAIN {} batch {} " + \
                 "fake_loss {:.5f} " + \
                 "real_loss {:.5f} " + \
                 "l1_loss {:.5f} " + \
                 "kld_loss {:.5f} " + \
                 "gan_loss {:.5f}").
                format(
                    self.batch_count,
                    i,
                    fake_loss.item(),
                    real_loss.item(),
                    l1_loss.item(),
                    kld_loss.item(),
                    gan_loss.item(),
                ))
            sys.stdout.flush()

        self.batch_count += 1
        return loss_meter

    def batch_test(self):
        self.generator.eval()
        self.discriminator.eval()
        loss_meter = Meter()

        for i, (states, cameras) in enumerate(self.test_loader):
            generated, means, logvars = self.generator(
                states.detach(), deterministic=True,
            )

            l1_loss, kld_loss, gan_loss = self._generator_loss(
                cameras, generated, means, logvars,
            )

            loss = (l1_loss * self.l1_loss_coeff +
                    kld_loss * self.kld_loss_coeff +
                    gan_loss * self.gan_loss_coeff)

            loss_meter.update(loss.item())

            print(
                ("TEST {} batch {} " + \
                 "l1_loss {:.5f} " + \
                 "kld_loss {:.5f} " + \
                 "gan_loss {:.5f}").
                format(
                    self.batch_count,
                    i,
                    l1_loss.item(),
                    kld_loss.item(),
                    gan_loss.item(),
                ))
            sys.stdout.flush()

        # Store policy if it did better.
        if loss_meter.avg < self.best_test_loss:
            self.best_test_loss = loss_meter.avg
            if self.save_dir:
                print("Saving models and optimizer: save_dir={} test_loss={}".format(
                    self.save_dir,
                    self.best_test_loss,
                ))
                torch.save(self.generator.state_dict(), self.save_dir + "/generator.pt")
                torch.save(self.generator_optimizer.state_dict(), self.save_dir + "/generator_optimizer.pt")
                torch.save(self.discriminator.state_dict(), self.save_dir + "/discriminator.pt")
                torch.save(self.discriminator_optimizer.state_dict(), self.save_dir + "/discriminator_optimizer.pt")

        return loss_meter
