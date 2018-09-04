import sys
import os
import cv2

import numpy as np

import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from pix2pix import Generator, Discriminator, VGG19, Encoder
from bdd100k import BDD100kSegInst, BDD100kBBox
from utils import Meter

# import pdb; pdb.set_trace()

class Pix2Pix:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device'))

        if config.get('dataset') == 'bdd100k_segmentation+instance':
            self.train_dataset = BDD100kSegInst(config, validation=False)
            self.test_dataset = BDD100kSegInst(config, validation=True)
        if config.get('dataset') == 'bdd100k_bbox':
            self.train_dataset = BDD100kBBox(config, validation=False)
            self.test_dataset = BDD100kBBox(config, validation=True)

        assert self.train_dataset is not None
        assert self.test_dataset is not None

        # self.encoder = Encoder(
        #     config, self.train_dataset.output_channel_count()
        # ).to(self.device)

        self.lcl_gen = LocalGenerator(
            config,
            self.train_dataset.input_channel_count(), # + self.encoder.feature_count(),
            self.train_dataset.output_channel_count(),
        ).to(self.device)
        self.gbl_gen = GlobalGenerator(
            config,
            self.train_dataset.input_channel_count(), # + self.encoder.feature_count(),
            self.train_dataset.output_channel_count(),
        ).to(self.device)

        self.save_dir = config.get('pix2pix_save_dir')
        self.load_dir = config.get('pix2pix_load_dir')
        self.tensorboard_log_dir = config.get('tensorboard_log_dir')

        if self.load_dir:
            self.lcl_gen.load_state_dict(
                torch.load(self.load_dir + "/lcl_gen.pt", map_location=self.device),
            )
            self.gbl_gen.load_state_dict(
                torch.load(self.load_dir + "/gbl_gen.pt", map_location=self.device),
            )

    def initialize_training(self):
        self.lcl_dis = Discriminator(
            self.config,
            self.train_dataset.input_channel_count() + \
            self.train_dataset.output_channel_count(),
        ).to(self.device)
        self.gbl_dis = Discriminator(
            self.config,
            self.train_dataset.input_channel_count() + \
            self.train_dataset.output_channel_count(),
        ).to(self.device)

        self.lcl_gen_optimizer = optim.Adam(
            self.lcl_gen.parameters(),
            lr=self.config.get('learning_rate'),
            betas=(self.config.get('adam_beta_1'), 0.999),
        )
        self.lcl_dis_optimizer = optim.Adam(
            self.lcl_dis.parameters(),
            lr=self.config.get('learning_rate'),
            betas=(self.config.get('adam_beta_1'), 0.999),
        )
        self.gbl_gen_optimizer = optim.Adam(
            self.gbl_gen.parameters(),
            lr=self.config.get('learning_rate'),
            betas=(self.config.get('adam_beta_1'), 0.999),
        )
        self.gbl_dis_optimizer = optim.Adam(
            self.glb_dis.parameters(),
            lr=self.config.get('learning_rate'),
            betas=(self.config.get('adam_beta_1'), 0.999),
        )
        self.vgg = VGG19().to(self.device)

        if self.load_dir:
            self.lcl_dis.load_state_dict(
                torch.load(self.load_dir + "/lcl_dis.pt", map_location=self.device),
            )
            self.lcl_dis_optimizer.load_state_dict(
                torch.load(self.load_dir + "/lcl_dis_optimizer.pt", map_location=self.device),
            )
            self.lcl_gen_optimizer.load_state_dict(
                torch.load(self.load_dir + "/lcl_gen_optimizer.pt", map_location=self.device),
            )
            self.gbl_dis.load_state_dict(
                torch.load(self.load_dir + "/gbl_dis.pt", map_location=self.device),
            )
            self.gbl_dis_optimizer.load_state_dict(
                torch.load(self.load_dir + "/gbl_dis_optimizer.pt", map_location=self.device),
            )
            self.gbl_gen_optimizer.load_state_dict(
                torch.load(self.load_dir + "/gbl_gen_optimizer.pt", map_location=self.device),
            )

        self.gan_layers_loss_coeff = self.config.get('gan_layers_loss_coeff')

        # We save the empty models initially if they were not loaded
        if self.load_dir is None and self.save_dir:
            torch.save(self.lcl_gen.state_dict(), self.save_dir + "/lcl_gen.pt")
            torch.save(self.lcl_gen_optimizer.state_dict(), self.save_dir + "/lcl_gen_optimizer.pt")
            torch.save(self.lcl_dis.state_dict(), self.save_dir + "/lcl_dis.pt")
            torch.save(self.lcl_dis_optimizer.state_dict(), self.save_dir + "/lcl_dis_optimizer.pt")
            torch.save(self.gbl_gen.state_dict(), self.save_dir + "/gbl_gen.pt")
            torch.save(self.gbl_gen_optimizer.state_dict(), self.save_dir + "/gbl_gen_optimizer.pt")
            torch.save(self.gbl_dis.state_dict(), self.save_dir + "/gbl_dis.pt")
            torch.save(self.gbl_dis_optimizer.state_dict(), self.save_dir + "/gbl_dis_optimizer.pt")

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.get('batch_size'),
            shuffle=True,
            num_workers=0,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.get('batch_size'),
            shuffle=False,
            num_workers=0,
        )

        self.tb_writer = None
        if self.tensorboard_log_dir:
            self.tb_writer = SummaryWriter(self.tensorboard_log_dir)

        self.iter = 0
        self.batch_count = 0

    def batch_train(self):
        self.lcl_gen.train()
        self.gbl_gen.train()
        self.lcl_dis.train()
        self.gbl_dis.train()

        lcl_loss_dis_fake_meter = Meter()
        lcl_loss_dis_real_meter = [Meter() for _ in config.get('gan_scale_count')]
        lcl_loss_gen_gan_meter = [Meter() for _ in config.get('gan_scale_count')]
        lcl_loss_gen_gan_feat_meter = Meter()
        lcl_loss_gen_vgg_feat_meter = Meter()

        gbl_loss_dis_fake_meter = Meter()
        gbl_loss_dis_real_meter = [Meter() for _ in config.get('gan_scale_count')]
        gbl_loss_gen_gan_meter = [Meter() for _ in config.get('gan_scale_count')]
        gbl_loss_gen_gan_feat_meter = Meter()

        for it, (labels, real_images) in enumerate(self.train_loader):
            lcl_labels = labels.to(self.device)

            gbl_labels, gbl_output, gbl_fake_images = self.gbl_gen(lcl_labels)
            _, lcl_fake_images = self.lcl_gen(lcl_labels, gbl_output)

            lcl_real_images = real_images
            gbl_real_images = self.gbl_gen.downsample(lcl_real_images)

            gbl_pred_dis_fake = self.gbl_dis(torch.cat((gbl_labels, gbl_fake_images.detach()), dim=1))
            gbl_pred_dis_real = self.gbl_dis(torch.cat((gbl_labels, gbl_real_images), dim=1))

            lcl_pred_dis_fake = self.lcl_dis(torch.cat((lcl_labels, lcl_fake_images.detach()), dim=1))
            lcl_pred_dis_real = self.lcl_dis(torch.cat((lcl_labels, lcl_real_images), dim=1))

            lcl_pred_gen_fake = self.lcl_dis(torch.cat((lcl_labels, lcl_fake_images), dim=1))
            gbl_pred_gen_fake = self.gbl_dis(torch.cat((gbl_labels, gbl_fake_images), dim=1))

            # Discriminator fake loss (fake images detached).
            lcl_loss_dis_fake = []
            for scale in lcl_pred_dis_fake:
                targ_dis_fake = torch.zeros(*scale[-1].size()).to(self.device).detach()
                output = scale[-1]
                lcl_loss_dis_fake.append(F.mse_loss(output, targ_dis_fake))
            gbl_loss_dis_fake = []
            for scale in gbl_pred_dis_fake:
                targ_dis_fake = torch.zeros(*scale[-1].size()).to(self.device).detach()
                output = scale[-1]
                gbl_loss_dis_fake.append(F.mse_loss(output, targ_dis_fake))

            # Discriminator real loss.
            lcl_loss_dis_real = []
            for scale in lcl_pred_dis_real:
                targ_dis_real = torch.ones(*scale[-1].size()).to(self.device).detach()
                output = scale[-1]
                lcl_loss_dis_real.append(F.mse_loss(output, targ_dis_real))
            gbl_loss_dis_real = []
            for scale in gbl_pred_dis_real:
                targ_dis_real = torch.ones(*scale[-1].size()).to(self.device).detach()
                output = scale[-1]
                gbl_loss_dis_real.append(F.mse_loss(output, targ_dis_real))

            # Generator GAN loss.
            lcl_loss_gen_gan = 0.0
            for scale in lcl_pred_gen_fake:
                targ_dis_real = torch.ones(*scale[-1].size()).to(self.device).detach()
                output = scale[-1]
                lcl_loss_gen_gan += F.mse_loss(output, targ_dis_real)
            gbl_loss_gen_gan = 0.0
            for scale in gbl_pred_gen_fake:
                targ_dis_real = torch.ones(*scale[-1].size()).to(self.device).detach()
                output = scale[-1]
                gbl_loss_gen_gan += F.mse_loss(output, targ_dis_real)

            # Generator GAN feature matching loss.
            layer_weights = 4.0 / (self.config.get('gan_layer_count') + 1)
            scale_weights = 1.0 / self.config.get('gan_scale_count')

            lcl_loss_gen_gan_feat = 0.0
            for i in range(len(lcl_pred_gen_fake)):
                for j in range(len(lcl_pred_gen_fake[i]) - 1):
                    lcl_loss_gen_gan_feat += layer_weights * scale_weights * \
                        F.l1_loss(
                            lcl_pred_gen_fake[i][j],
                            lcl_pred_dis_real[i][j].detach(),
                        )
            gbl_loss_gen_gan_feat = 0.0
            for i in range(len(gbl_pred_gen_fake)):
                for j in range(len(gbl_pred_gen_fake[i]) - 1):
                    gbl_loss_gen_gan_feat += layer_weights * scale_weights * \
                        F.l1_loss(
                            gbl_pred_gen_fake[i][j],
                            gbl_pred_dis_real[i][j].detach(),
                        )

            # Generator VGG feature matching loss (lcl only).
            lcl_loss_gen_vgg_feat = 0.0
            vgg_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
            lcl_out_vgg_fake = self.vgg(lcl_fake_images)
            lcl_out_vgg_real = self.vgg(lcl_real_images)
            for i in range(len(vgg_weights)):
                lcl_loss_gen_vgg_feat += vgg_weights[i] * \
                    F.l1_loss(
                        lcl_out_vgg_fake[i],
                        lcl_out_vgg_real[i],
                    )

            # Generator backpropagation.
            self.gbl_gen_optimizer.zero_grad()
            (
                gbl_loss_gen_gan +
                gbl_loss_gen_gan_feat * self.gan_layers_loss_coeff
            ).backward()
            self.gbl_gen_optimizer.step()

            self.lcl_gen_optimizer.zero_grad()
            (
                lcl_loss_gen_gan +
                lcl_loss_gen_gan_feat * self.gan_layers_loss_coeff +
                lcl_loss_gen_vgg_feat * self.gan_layers_loss_coeff
            ).backward()
            self.lcl_gen_optimizer.step()

            # Discriminator backpropagation.
            self.lcl_dis_optimizer.zero_grad()
            (
                sum(lcl_loss_dis_real) +
                sum(lcl_loss_dis_fake)
            ).backward()
            self.lcl_dis_optimizer.step()

            self.gbl_dis_optimizer.zero_grad()
            (
                sum(gbl_loss_dis_real) +
                sum(gbl_loss_dis_fake)
            ).backward()
            self.gbl_dis_optimizer.step()

            for i in range(config.get('gan_scale_count')):
                lcl_loss_dis_fake_meter[i].update(lcl_loss_dis_fake[i].item())
                lcl_loss_dis_real_meter[i].update(lcl_loss_dis_real[i].item())
                gbl_loss_dis_fake_meter[i].update(gbl_loss_dis_fake[i].item())
                gbl_loss_dis_real_meter[i].update(gbl_loss_dis_real[i].item())
            lcl_loss_gen_gan_meter.update(lcl_loss_gen_gan.item())
            lcl_loss_gen_gan_feat_meter.update(lcl_loss_gen_gan_feat.item())
            lcl_loss_gen_vgg_feat_meter.update(lcl_loss_gen_vgg_feat.item())
            gbl_loss_gen_gan_meter.update(gbl_loss_gen_gan.item())
            gbl_loss_gen_gan_feat_meter.update(gbl_loss_gen_gan_feat.item())

            self.iter += 1

        print("TRAIN {}".format(self.batch_count))
        sys.stdout.flush()

        if self.tb_writer is not None:
            for i in range(config.get('gan_scale_count')):
                self.tb_writer.add_scalar("train/loss/lcl/dis/fake/{}".format(i), lcl_loss_dis_fake_meter[i].avg, self.batch_count)
                self.tb_writer.add_scalar("train/loss/lcl/dis/real".format(i), lcl_loss_dis_real_meter[i].avg, self.batch_count)
                self.tb_writer.add_scalar("train/loss/gbl/dis/fake/{}".format(i), gbl_loss_dis_fake_meter[i].avg, self.batch_count)
                self.tb_writer.add_scalar("train/loss/gbl/dis/real".format(i), gbl_loss_dis_real_meter[i].avg, self.batch_count)
            self.tb_writer.add_scalar('train/loss/lcl/gen/gan', lcl_loss_gen_gan_meter.avg, self.batch_count)
            self.tb_writer.add_scalar('train/loss/lcl/gen/gan_feat', lcl_loss_gen_gan_feat_meter.avg, self.batch_count)
            self.tb_writer.add_scalar('train/loss/lcl/gen/vgg_feat', lcl_loss_gen_vgg_feat_meter.avg, self.batch_count)
            self.tb_writer.add_scalar('train/loss/gbl/gen/gan', gbl_loss_gen_gan_meter.avg, self.batch_count)
            self.tb_writer.add_scalar('train/loss/gbl/gen/gan_feat', gbl_loss_gen_gan_feat_meter.avg, self.batch_count)

        self.batch_count += 1

    def batch_test(self):
        self.lcl_gen.train()
        self.gbl_gen.train()

        for it, (labels, real_images) in enumerate(self.test_loader):
            lcl_labels = labels.to(self.device)
            _, gbl_output, gbl_fake_images = self.gbl_gen(lcl_labels)
            _, lcl_fake_images = self.lcl_gen(lcl_labels, gbl_output)

            lcl_real_images = real_images
            gbl_real_images = self.gbl_gen.downsample(lcl_real_images)

            if self.tb_writer is not None:
                lcl_grid = torchvision.utils.make_grid([
                    # ((labels[0].cpu() + 1.0) * 127.5).to(torch.uint8),
                    ((lcl_real_images[0].cpu() + 1.0) * 127.5).to(torch.uint8),
                    ((lcl_fake_images[0].cpu() + 1.0) * 127.5).to(torch.uint8),
                ])
                self.tb_writer.add_image(
                    'test/lcl/fake_images/{}'.format(it),
                    np.flip(lcl_grid.numpy(), axis=0),
                    self.batch_count,
                )
                grid = torchvision.utils.make_grid([
                    # ((labels[0].cpu() + 1.0) * 127.5).to(torch.uint8),
                    ((gbl_real_images[0].cpu() + 1.0) * 127.5).to(torch.uint8),
                    ((gbl_fake_images[0].cpu() + 1.0) * 127.5).to(torch.uint8),
                ])
                self.tb_writer.add_image(
                    'test/gbl/fake_images/{}'.format(it),
                    np.flip(lcl_grid.numpy(), axis=0),
                    self.batch_count,
                )

        if self.save_dir:
            print("Saving models and optimizers: iter={}".format(self.batch_count))
            torch.save(self.lcl_gen.state_dict(), self.save_dir + "/lcl_gen.pt")
            torch.save(self.lcl_dis.state_dict(), self.save_dir + "/lcl_dis.pt")
            torch.save(self.lcl_gen_optimizer.state_dict(), self.save_dir + "/lcl_gen_optimizer.pt")
            torch.save(self.lcl_dis_optimizer.state_dict(), self.save_dir + "/lcl_dis_optimizer.pt")
            torch.save(self.gbl_gen.state_dict(), self.save_dir + "/gbl_gen.pt")
            torch.save(self.gbl_dis.state_dict(), self.save_dir + "/gbl_dis.pt")
            torch.save(self.gbl_gen_optimizer.state_dict(), self.save_dir + "/gbl_gen_optimizer.pt")
            torch.save(self.gbl_dis_optimizer.state_dict(), self.save_dir + "/gbl_dis_optimizer.pt")

