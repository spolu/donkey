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

# import pdb; pdb.set_trace()

class Synthetic:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device'))

        self.vae = VAE(config).to(self.device)
        self.stl = STL(config).to(self.device)

        self.save_dir = config.get('synthetic_save_dir')
        self.load_dir = config.get('synthetic_load_dir')
        self.frame_stack_size = config.get('frame_stack_size')

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

    def generate(self, states):
        self.vae.eval()
        self.stl.eval()

        assert len(states) == self.frame_stack_size

        stl_latent = self.stl(
            torch.cat([
                torch.from_numpy(state.vector())
                for state in states
            ]).unsqueeze(0).float().to(self.device)
        )
        generated = self.vae.decode(stl_latent)
        camera = torch.cat(
            (
                torch.zeros(
                    1, self.frame_stack_size, 50, generated.size(3),
                ).to(self.device),
                generated,
            ),
            dim=2,
        ).detach()

        return (camera * 255.0).cpu().numpy()

    def _generator_capture_loader(self, items):
        states = [
            torch.from_numpy(State(
                item['simulation_track_randomization'],
                item['simulation_position'],
                item['simulation_velocity'],
                item['simulation_angular_velocity'],
                item['simulation_track_coordinates'],
                item['simulation_track_angle'],
            ).vector()).float().to(self.device)
            for item in items
        ]
        state = torch.cat(states)

        cameras = [
            torch.from_numpy(
                cv2.imdecode(
                    np.fromstring(item['camera'], np.uint8),
                    cv2.IMREAD_GRAYSCALE,
                ) / 255.0
            ).float().to(self.device).unsqueeze(0)
            for item in items
        ]
        camera = torch.cat(cameras)

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
        self.stl_e2e_l1_loss_coeff = self.config.get('stl_e2e_l1_loss_coeff')
        self.stl_e2e_mse_loss_coeff = self.config.get('stl_e2e_mse_loss_coeff')
        self.stl_e2e_bce_loss_coeff = self.config.get('stl_e2e_bce_loss_coeff')

        self.vae_epoch_count = self.config.get('vae_epoch_count')
        self.stl_epoch_count = self.config.get('stl_epoch_count')

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

        # We save the empty models initially if they were not loaded
        if self.load_dir is None and self.save_dir:
            torch.save(self.vae.state_dict(), self.save_dir + "/vae.pt")
            torch.save(self.vae_optimizer.state_dict(), self.save_dir + "/vae_optimizer.pt")
            torch.save(self.stl.state_dict(), self.save_dir + "/stl.pt")
            torch.save(self.stl_optimizer.state_dict(), self.save_dir + "/stl_optimizer.pt")
            torch.save(self.discriminator.state_dict(), self.save_dir + "/discriminator.pt")
            torch.save(self.discriminator_optimizer.state_dict(), self.save_dir + "/discriminator_optimizer.pt")

        self.train_capture_set = CaptureSet(
            train_capture_set_dir,
            loader=self._generator_capture_loader,
            frame_size=self.frame_stack_size,
        )
        self.test_capture_set = CaptureSet(
            test_capture_set_dir,
            loader=self._generator_capture_loader,
            frame_size=self.frame_stack_size,
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
        self.best_stl_loss = 9999.0
        self.best_vae_loss = 9999.0

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

    def _stl_loss(self, vae_latent, stl_latent, camera, generated):
        mse_loss = F.mse_loss(
            stl_latent, vae_latent.detach(),
        )

        e2e_l1_loss = F.l1_loss(
            generated, camera.detach(),
        )
        e2e_mse_loss = F.mse_loss(
            generated, camera.detach(),
        )
        e2e_bce_loss = F.binary_cross_entropy(
            generated, camera.detach(),
        )

        return mse_loss, e2e_l1_loss, e2e_mse_loss, e2e_bce_loss

    def batch_train(self):
        self.vae.train()
        self.discriminator.train()
        self.vae.set_requires_grad(True)

        for epoch in range(self.vae_epoch_count):
            for i, (state, camera) in enumerate(self.train_loader):
                vae_latent, encoded, vae_mean, vae_logvar = self.vae(
                    camera.detach(), deterministic=False,
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

                print(
                    ("TRAIN_VAE {} batch {}-{} " + \
                     "fake_loss {:.5f} " + \
                     "real_loss {:.5f} " + \
                     "vae_l1_loss {:.5f} " + \
                     "vae_mse_loss {:.5f} " + \
                     "vae_bce_loss {:.5f} " + \
                     "vae_kld_loss {:.5f} " + \
                     "vae_gan_loss {:.5f}").
                    format(
                        self.batch_count,
                        epoch, i,
                        fake_loss.item(),
                        real_loss.item(),
                        vae_l1_loss.item(),
                        vae_mse_loss.item(),
                        vae_bce_loss.item(),
                        vae_kld_loss.item(),
                        vae_gan_loss.item(),
                    ))
                sys.stdout.flush()

        self.stl.train()
        self.vae.set_requires_grad(False)

        for epoch in range(self.stl_epoch_count):
            for i, (state, camera) in enumerate(self.train_loader):
                vae_latent, encoded, vae_mean, vae_logvar = self.vae(
                    camera.detach(), deterministic=False,
                )
                stl_latent = self.stl(
                    state.detach()
                )
                generated = self.vae.decode(stl_latent.detach())

                # STL pass.
                self.stl_optimizer.zero_grad()

                stl_mse_loss, stl_e2e_l1_loss, stl_e2e_mse_loss, stl_e2e_bce_loss = self._stl_loss(
                    vae_latent, stl_latent, camera, generated,
                )

                stl_loss = (
                    stl_mse_loss * self.stl_mse_loss_coeff +
                    stl_e2e_l1_loss * self.stl_e2e_l1_loss_coeff +
                    stl_e2e_mse_loss * self.stl_e2e_mse_loss_coeff +
                    stl_e2e_bce_loss * self.stl_e2e_bce_loss_coeff
                )
                stl_loss.backward()

                self.stl_optimizer.step()

                if i % 100 == 0:
                    if self.save_dir:
                        cv2.imwrite(
                            os.path.join(self.save_dir, '{}_synthetic_original.jpg'.format(i)),
                            (255 * camera[0][0].to('cpu')).detach().numpy(),
                        )
                        cv2.imwrite(
                            os.path.join(self.save_dir, '{}_synthetic_encoded.jpg'.format(i)),
                            (255 * encoded[0][0].to('cpu')).detach().numpy(),
                        )
                        cv2.imwrite(
                            os.path.join(self.save_dir, '{}_synthetic_generated.jpg'.format(i)),
                            (255 * generated[0][0].to('cpu')).detach().numpy(),
                        )

                print(
                    ("TRAIN_STL {} batch {}-{} " + \
                     "stl_mse_loss {:.5f} " + \
                     "stl_e2e_l1_loss {:.5f} " + \
                     "stl_e2e_mse_loss {:.5f} " + \
                     "stl_e2e_bce_loss {:.5f}").
                    format(
                        self.batch_count,
                        epoch, i,
                        stl_mse_loss.item(),
                        stl_e2e_l1_loss.item(),
                        stl_e2e_mse_loss.item(),
                        stl_e2e_bce_loss.item(),
                    ))
                sys.stdout.flush()


        self.batch_count += 1

    def batch_test(self):
        self.vae.eval()
        self.stl.eval()
        self.discriminator.eval()
        stl_e2e_l1_loss_meter = Meter()
        stl_loss_meter = Meter()
        vae_loss_meter = Meter()

        for i, (state, camera) in enumerate(self.test_loader):
            vae_latent, encoded, vae_mean, vae_logvar = self.vae(
                camera.detach(), deterministic=True,
            )
            stl_latent = self.stl(
                state.detach()
            )
            generated = self.vae.decode(stl_latent.detach())

            # VAE loss.
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

            vae_loss_meter.update(vae_loss.item())

            # E2E (STL) loss.
            stl_mse_loss, stl_e2e_l1_loss, stl_e2e_mse_loss, stl_e2e_bce_loss = self._stl_loss(
                vae_latent, stl_latent, camera, generated,
            )

            stl_loss = (
                stl_mse_loss * self.stl_mse_loss_coeff +
                stl_e2e_l1_loss * self.stl_e2e_l1_loss_coeff +
                stl_e2e_mse_loss * self.stl_e2e_mse_loss_coeff +
                stl_e2e_bce_loss * self.stl_e2e_bce_loss_coeff
            )

            stl_loss_meter.update(stl_loss.item())
            stl_e2e_l1_loss_meter.update(stl_e2e_l1_loss.item())

            print(
                ("TEST {} batch {} " + \
                 "vae_l1_loss {:.5f} " + \
                 "vae_mse_loss {:.5f} " + \
                 "vae_bce_loss {:.5f} " + \
                 "vae_kld_loss {:.5f} " + \
                 "vae_gan_loss {:.5f} " + \
                 "stl_mse_loss {:.5f} " + \
                 "stl_e2e_l1_loss {:.5f} " + \
                 "stl_e2e_mse_loss {:.5f} " + \
                 "stl_e2e_bce_loss {:.5f}").
                format(
                    self.batch_count,
                    i,
                    vae_l1_loss.item(),
                    vae_mse_loss.item(),
                    vae_bce_loss.item(),
                    vae_kld_loss.item(),
                    vae_gan_loss.item(),
                    stl_mse_loss.item(),
                    stl_e2e_l1_loss.item(),
                    stl_e2e_mse_loss.item(),
                    stl_e2e_bce_loss.item(),
                ))
            sys.stdout.flush()

        # Store policy if it did better.
        if self.save_dir:
            if self.stl_epoch_count > 0 and self.best_stl_loss > stl_loss_meter.avg:
                self.best_stl_loss = stl_loss_meter.avg
                print(
                    ("Saving STL models and optimizer: episode={} " + \
                     "stl_loss={} " + \
                     "stl_e2e_l1_loss={}").
                    format(
                        self.batch_count,
                        stl_loss_meter.avg,
                        stl_e2e_l1_loss_meter.avg,
                    ))
                torch.save(self.stl.state_dict(), self.save_dir + "/stl.pt")
                torch.save(self.stl_optimizer.state_dict(), self.save_dir + "/stl_optimizer.pt")
            if self.vae_epoch_count > 0 and self.best_vae_loss > vae_loss_meter.avg:
                self.best_vae_loss = vae_loss_meter.avg
                print(
                    ("Saving VAE models and optimizer: episode={} " +
                     "vae_loss={}")
                    .format(
                        self.batch_count,
                        self.best_vae_loss,
                    ))
                torch.save(self.vae.state_dict(), self.save_dir + "/vae.pt")
                torch.save(self.vae_optimizer.state_dict(), self.save_dir + "/vae_optimizer.pt")
                torch.save(self.discriminator.state_dict(), self.save_dir + "/discriminator.pt")
                torch.save(self.discriminator_optimizer.state_dict(), self.save_dir + "/discriminator_optimizer.pt")

        return stl_loss_meter
