import cv2
import sys
import os

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from synthetic import Generator, Discriminator, State, VAE
from capture import CaptureSet
from utils import Meter
from reinforce import InputFilter

# import pdb; pdb.set_trace()

class Synthetic:
    def __init__(self, config, save_dir=None, load_dir=None):
        self.config = config
        self.device = torch.device(config.get('device'))
        self.input_filter = InputFilter(config)

        self.vae = VAE(config).to(self.device)

        self.save_dir = save_dir
        self.load_dir = load_dir

        if self.load_dir:
            if config.get('device') != 'cpu':
                self.vae.load_state_dict(
                    torch.load(self.load_dir + "/vae.pt"),
                )
            else:
                self.vae.load_state_dict(
                    torch.load(self.load_dir + "/vae.pt", map_location='cpu'),
                )

    # def generate(self, state):
    #     self.generator.eval()
    #     generated, _, _ = self.generator(
    #         torch.from_numpy(
    #             state.vector()
    #         ).float().to(self.device).unsqueeze(0).detach(),
    #         deterministic=True,
    #     )
    #     return generated * 255.0

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

        self.vae_optimizer = optim.Adam(
            self.vae.parameters(),
            self.config.get('vae_learning_rate'),
        )
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            self.config.get('discriminator_learning_rate'),
        )
        self.l1_loss_coeff = self.config.get('l1_loss_coeff')
        self.mse_loss_coeff = self.config.get('mse_loss_coeff')
        self.bce_loss_coeff = self.config.get('bce_loss_coeff')
        self.kld_loss_coeff = self.config.get('kld_loss_coeff')
        self.gan_loss_coeff = self.config.get('gan_loss_coeff')

        if self.load_dir:
            if self.config.get('device') != 'cpu':
                self.vae_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/vae_optimizer.pt"),
                )
                self.discriminator.load_state_dict(
                    torch.load(self.load_dir + "/discriminator.pt"),
                )
                self.discriminator_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/discriminator_optimizer.pt"),
                )
            else:
                self.vae_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/vae_optimizer.pt", map_location='cpu'),
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

    def _discriminator_loss(self, camera, encoded):
        pred_fake = self.discriminator(encoded.detach())
        fake_loss = F.binary_cross_entropy(
            pred_fake,
            torch.zeros(*pred_fake.size()).to(self.device),
        )

        pred_real = self.discriminator(camera.detach())
        real_loss = F.binary_cross_entropy(
            pred_real,
            torch.ones(*pred_real.size()).to(self.device),
        )

        return fake_loss, real_loss

    def _vae_loss(self, camera, encoded, mean, logvar):
        # Do not detach as we want the gradients to flow from D to G.
        pred_gan = self.discriminator(encoded)
        gan_loss = F.binary_cross_entropy(
            pred_gan,
            torch.ones(*pred_gan.size()).to(self.device),
        )

        l1_loss = F.l1_loss(
            encoded, camera.detach(),
        )
        mse_loss = F.mse_loss(
            encoded, camera.detach(),
        )
        bce_loss = F.binary_cross_entropy(
            encoded, camera.detach(),
        )
        kld_loss = -0.5 * torch.sum(
            1 + logvar - mean.pow(2) - logvar.exp()
        )
        kld_loss /= encoded.size(0) * encoded.size(1) * encoded.size(2)

        return l1_loss, mse_loss, bce_loss, kld_loss, gan_loss

    def batch_train(self):
        self.vae.train()
        self.discriminator.train()
        loss_meter = Meter()

        for i, (state, camera) in enumerate(self.train_loader):
            latent, encoded, mean, logvar = self.vae(
                camera.detach(), deterministic=True,
            )

            # Discriminator pass.
            self.discriminator.set_requires_grad(True)
            self.discriminator_optimizer.zero_grad()

            fake_loss, real_loss = self._discriminator_loss(
                camera, encoded,
            )

            discriminator_loss = (
                fake_loss * 0.5 +
                real_loss * 0.5
            )
            discriminator_loss.backward()

            self.discriminator_optimizer.step()

            # VAE pass.
            self.discriminator.set_requires_grad(False)
            self.vae_optimizer.zero_grad()

            l1_loss, mse_loss, bce_loss, kld_loss, gan_loss = self._vae_loss(
                camera, encoded, mean, logvar,
            )

            vae_loss = (
                l1_loss * self.l1_loss_coeff +
                mse_loss * self.mse_loss_coeff +
                bce_loss * self.bce_loss_coeff +
                kld_loss * self.kld_loss_coeff +
                gan_loss * self.gan_loss_coeff
            )
            vae_loss.backward()

            self.vae_optimizer.step()

            loss_meter.update(vae_loss.item())

            if i % 1000 == 0:
                if self.save_dir:
                    cv2.imwrite(
                        os.path.join(self.save_dir, '{}_synthetic_original.jpg'.format(i)),
                        (255 * camera[0].squeeze(0).to('cpu')).detach().numpy(),
                    )
                    cv2.imwrite(
                        os.path.join(self.save_dir, '{}_synthetic_encoded.jpg'.format(i)),
                        (255 * encoded[0].squeeze(0).to('cpu')).detach().numpy(),
                    )

            print(
                ("TRAIN {} batch {} " + \
                 "fake_loss {:.5f} " + \
                 "real_loss {:.5f} " + \
                 "l1_loss {:.5f} " + \
                 "mse_loss {:.5f} " + \
                 "bce_loss {:.5f} " + \
                 "kld_loss {:.5f} " + \
                 "gan_loss {:.5f}").
                format(
                    self.batch_count,
                    i,
                    fake_loss.item(),
                    real_loss.item(),
                    l1_loss.item(),
                    mse_loss.item(),
                    bce_loss.item(),
                    kld_loss.item(),
                    gan_loss.item(),
                ))
            sys.stdout.flush()

        self.batch_count += 1
        return loss_meter

    def batch_test(self):
        self.vae.eval()
        self.discriminator.eval()
        loss_meter = Meter()

        for i, (state, camera) in enumerate(self.test_loader):
            latent, encoded, mean, logvar = self.vae(
                camera.detach(), deterministic=True,
            )

            l1_loss, mse_loss, bce_loss, kld_loss, gan_loss = self._vae_loss(
                camera, encoded, mean, logvar,
            )

            loss = (
                l1_loss * self.l1_loss_coeff +
                mse_loss * self.mse_loss_coeff +
                bce_loss * self.bce_loss_coeff +
                kld_loss * self.kld_loss_coeff +
                gan_loss * self.gan_loss_coeff
            )

            loss_meter.update(loss.item())

            print(
                ("TEST {} batch {} " + \
                 "l1_loss {:.5f} " + \
                 "mse_loss {:.5f} " + \
                 "kld_loss {:.5f} " + \
                 "gan_loss {:.5f}").
                format(
                    self.batch_count,
                    i,
                    l1_loss.item(),
                    mse_loss.item(),
                    bce_loss.item(),
                    kld_loss.item(),
                    gan_loss.item(),
                ))
            sys.stdout.flush()

        # Store policy if it did better.
        if self.save_dir:
            print("Saving models and optimizer: save_dir={}".format(
                self.save_dir,
            ))
            torch.save(self.vae.state_dict(), self.save_dir + "/vae.pt")
            torch.save(self.vae_optimizer.state_dict(), self.save_dir + "/vae_optimizer.pt")
            torch.save(self.discriminator.state_dict(), self.save_dir + "/discriminator.pt")
            torch.save(self.discriminator_optimizer.state_dict(), self.save_dir + "/discriminator_optimizer.pt")

        return loss_meter
