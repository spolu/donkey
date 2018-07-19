import cv2
import sys
import os

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from synthetic import Discriminator, State, VAE, STL
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
        self.stl = STL(config).to(self.device)

        self.save_dir = save_dir
        self.load_dir = load_dir

        if self.load_dir:
            if config.get('device') != 'cpu':
                self.vae.load_state_dict(
                    torch.load(self.load_dir + "/vae.pt"),
                )
                self.stl.load_state_dict(
                    torch.load(self.load_dir + "/stl.pt"),
                )
            else:
                self.vae.load_state_dict(
                    torch.load(self.load_dir + "/vae.pt", map_location='cpu'),
                )
                self.stl.load_state_dict(
                    torch.load(self.load_dir + "/stl.pt", map_location='cpu'),
                )

    def generate(self, state):
        self.vae.eval()
        self.stl.eval()

        stl_latent = self.stl(
            torch.from_numpy(state.vector()).float().to(self.device),
            deterministic=True,
        )
        generated = self.vae.decode(stl_latent)

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

        self.vae_optimizer = optim.Adam(
            self.vae.parameters(),
            self.config.get('vae_learning_rate'),
        )
        self.stl_optimizer = optim.Adam(
            self.stl.parameters(),
            self.config.get('stl_learning_rate'),
        )
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            self.config.get('discriminator_learning_rate'),
        )
        self.vae_l1_loss_coeff = self.config.get('vae_l1_loss_coeff')
        self.vae_mse_loss_coeff = self.config.get('vae_mse_loss_coeff')
        self.vae_bce_loss_coeff = self.config.get('vae_bce_loss_coeff')
        self.vae_kld_loss_coeff = self.config.get('vae_kld_loss_coeff')
        self.vae_gan_loss_coeff = self.config.get('vae_gan_loss_coeff')

        self.stl_mse_loss_coeff = self.config.get('stl_mse_loss_coeff')

        if self.load_dir:
            if self.config.get('device') != 'cpu':
                self.vae_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/vae_optimizer.pt"),
                )
                self.stl_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/stl_optimizer.pt"),
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
                self.stl_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/stl_optimizer.pt", map_location='cpu'),
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

    def _vae_loss(self, camera, encoded, vae_mean, vae_logvar):
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
            1 + vae_logvar - vae_mean.pow(2) - vae_logvar.exp()
        )
        kld_loss /= encoded.size(0) * encoded.size(1) * encoded.size(2)

        return l1_loss, mse_loss, bce_loss, kld_loss, gan_loss

    def _stl_loss(self, vae_latent, stl_latent):
        mse_loss = F.mse_loss(
            stl_latent, vae_latent.detach(),
        )

        return mse_loss

    def batch_train(self):
        self.vae.train()
        self.stl.train()
        self.discriminator.train()

        for i, (state, camera) in enumerate(self.train_loader):
            vae_latent, encoded, vae_mean, vae_logvar = self.vae(
                camera.detach(), deterministic=False,
            )
            stl_latent = self.stl(
                state.detach()
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
            self.vae_optimizer.zero_grad()

            vae_l1_loss, vae_mse_loss, vae_bce_loss, vae_kld_loss, vae_gan_loss = self._vae_loss(
                camera, encoded, vae_mean, vae_logvar,
            )

            vae_loss = (
                vae_l1_loss * self.vae_l1_loss_coeff +
                vae_mse_loss * self.vae_mse_loss_coeff +
                vae_bce_loss * self.vae_bce_loss_coeff +
                vae_kld_loss * self.vae_kld_loss_coeff +
                vae_gan_loss * self.vae_gan_loss_coeff
            )
            vae_loss.backward()

            self.vae_optimizer.step()

            # STL pass.
            self.stl_optimizer.zero_grad()

            stl_mse_loss = self._stl_loss(
                vae_latent, stl_latent,
            )

            stl_loss = (
                stl_mse_loss * self.stl_mse_loss_coeff
            )
            stl_loss.backward()

            self.stl_optimizer.step()

            generated = self.vae.decode(stl_latent.detach())

            e2e_mse_loss = F.mse_loss(
                generated, camera.detach(),
            )

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
                    cv2.imwrite(
                        os.path.join(self.save_dir, '{}_synthetic_generated.jpg'.format(i)),
                        (255 * generated[0].squeeze(0).to('cpu')).detach().numpy(),
                    )

            print(
                ("TRAIN {} batch {} " + \
                 "fake_loss {:.5f} " + \
                 "real_loss {:.5f} " + \
                 "vae_l1_loss {:.5f} " + \
                 "vae_mse_loss {:.5f} " + \
                 "vae_bce_loss {:.5f} " + \
                 "vae_kld_loss {:.5f} " + \
                 "vae_gan_loss {:.5f} " + \
                 "stl_mse_loss {:.5f} " + \
                 "e2e_mse_loss {:.5f}").
                format(
                    self.batch_count,
                    i,
                    fake_loss.item(),
                    real_loss.item(),
                    vae_l1_loss.item(),
                    vae_mse_loss.item(),
                    vae_bce_loss.item(),
                    vae_kld_loss.item(),
                    vae_gan_loss.item(),
                    stl_mse_loss.item(),
                    e2e_mse_loss.item(),
                ))
            sys.stdout.flush()

        self.batch_count += 1

    def batch_test(self):
        self.vae.eval()
        self.stl.eval()
        self.discriminator.eval()
        loss_meter = Meter()

        for i, (state, camera) in enumerate(self.test_loader):
            vae_latent, encoded, vae_mean, vae_logvar = self.vae(
                camera.detach(), deterministic=True,
            )
            stl_latent = self.stl(
                state.detach()
            )
            generated = self.vae.decode(stl_latent.detach())

            e2e_mse_loss = F.mse_loss(
                generated, camera.detach(),
            )

            loss_meter.update(e2e_mse_loss.item())

            print(
                ("TEST {} batch {} " + \
                 "e2e_mse_loss {:.5f}").
                format(
                    self.batch_count,
                    i,
                    e2e_mse_loss.item(),
                ))
            sys.stdout.flush()

        # Store policy if it did better.
        if self.save_dir:
            if self.best_test_loss > loss_meter.avg:
                self.best_test_loss = loss_meter.avg
                print("Saving models and optimizer: save_dir={} test_loss={}".format(
                    self.save_dir, self.best_test_loss,
                ))
                torch.save(self.vae.state_dict(), self.save_dir + "/vae.pt")
                torch.save(self.vae_optimizer.state_dict(), self.save_dir + "/vae_optimizer.pt")
                torch.save(self.stl.state_dict(), self.save_dir + "/stl.pt")
                torch.save(self.stl_optimizer.state_dict(), self.save_dir + "/stl_optimizer.pt")
                torch.save(self.discriminator.state_dict(), self.save_dir + "/discriminator.pt")
                torch.save(self.discriminator_optimizer.state_dict(), self.save_dir + "/discriminator_optimizer.pt")

        return loss_meter
