import sys
import os

import numpy as np

import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from pix2pix import Generator, Discriminator, VGG19, Encoder
from bdd100k import BDD100kSegInst
from utils import Meter

# import pdb; pdb.set_trace()

class Pix2Pix:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device'))

        if config.get('dataset') == 'bdd100k_segmentation+instance':
            self.train_dataset = BDD100kSegInst(config, validation=False)
            self.test_dataset = BDD100kSegInst(config, validation=True)

        assert self.train_dataset is not None
        assert self.test_dataset is not None

        # self.encoder = Encoder(
        #     config, self.train_dataset.output_channel_count()
        # ).to(self.device)

        self.generator = Generator(
            config,
            self.train_dataset.input_channel_count(), # + self.encoder.feature_count(),
            self.train_dataset.output_channel_count(),
        ).to(self.device)

        self.save_dir = config.get('pix2pix_save_dir')
        self.load_dir = config.get('pix2pix_load_dir')
        self.tensorboard_log_dir = config.get('tensorboard_log_dir')

        if self.load_dir:
            if config.get('device') != 'cpu':
                self.generator.load_state_dict(
                    torch.load(self.load_dir + "/generator.pt"),
                )
            else:
                self.generator.load_state_dict(
                    torch.load(self.load_dir + "/generator.pt", map_location='cpu'),
                )

    def generate(self, labels):
        self.generator.eval()

        assert len(labels) == self.train_dataset.input_channel_count()

        generated = self.generator(
            labels.unsqueeze(0).float().to(self.device),
        )

        return self.train_dataset.postprocess(generated).cpu().numpy()

    def initialize_training(self):
        self.discriminator = Discriminator(
            self.config,
            self.train_dataset.input_channel_count() + \
            self.train_dataset.output_channel_count(),
        ).to(self.device)

        self.gen_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.config.get('learning_rate'),
            betas=(self.config.get('adam_beta_1'), 0.999),
        )
        self.dis_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.get('learning_rate'),
            betas=(self.config.get('adam_beta_1'), 0.999),
        )
        self.vgg = VGG19().to(self.device)

        if self.load_dir:
            if self.config.get('device') != 'cpu':
                self.discriminator.load_state_dict(
                    torch.load(self.load_dir + "/discriminator.pt"),
                )
                self.dis_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/dis_optimizer.pt"),
                )
                self.gen_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/gen_optimizer.pt"),
                )
            else:
                self.discriminator.load_state_dict(
                    torch.load(self.load_dir + "/discriminator.pt", map_location='cpu'),
                )
                self.dis_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/dis_optimizer.pt", map_location='cpu'),
                )
                self.gen_optimizer.load_state_dict(
                    torch.load(self.load_dir + "/gen_optimizer.pt", map_location='cpu'),
                )

        self.gan_layers_loss_coeff = self.config.get('gan_layers_loss_coeff')

        # We save the empty models initially if they were not loaded
        if self.load_dir is None and self.save_dir:
            torch.save(self.generator.state_dict(), self.save_dir + "/generator.pt")
            torch.save(self.gen_optimizer.state_dict(), self.save_dir + "/gen_optimizer.pt")
            torch.save(self.discriminator.state_dict(), self.save_dir + "/discriminator.pt")
            torch.save(self.dis_optimizer.state_dict(), self.save_dir + "/dis_optimizer.pt")

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
        self.generator.train()
        self.discriminator.train()

        loss_dis_fake_meter = Meter()
        loss_dis_real_meter = Meter()
        loss_gen_gan_meter = Meter()
        loss_gen_gan_feat_meter = Meter()
        loss_gen_vgg_feat_meter = Meter()

        for it, (labels, real_images) in enumerate(self.train_loader):
            fake_images = self.generator(labels.to(self.device))

            pred_dis_fake = self.discriminator(torch.cat((labels, fake_images.detach()), dim=1))
            pred_dis_real = self.discriminator(torch.cat((labels, real_images), dim=1))

            # Discriminator fake loss (fake images detached).
            loss_dis_fake = 0.0
            for scale in pred_dis_fake:
                targ_dis_fake = torch.zeros(*scale[-1].size()).to(self.device).detach()
                output = scale[-1]
                loss_dis_fake += F.mse_loss(output, targ_dis_fake)

            # Discriminator real loss.
            loss_dis_real = 0.0
            for scale in pred_dis_real:
                targ_dis_real = torch.ones(*scale[-1].size()).to(self.device).detach()
                output = scale[-1]
                loss_dis_real += F.mse_loss(output, targ_dis_real)


            pred_gen_fake = self.discriminator(torch.cat((labels, fake_images), dim=1))

            # Generator GAN loss.
            loss_gen_gan = 0.0
            for scale in pred_gen_fake:
                targ_dis_real = torch.ones(*scale[-1].size()).to(self.device).detach()
                output = scale[-1]
                loss_gen_gan += F.mse_loss(output, targ_dis_real)

            # Generator GAN feature matching loss.
            loss_gen_gan_feat = 0.0
            layer_weights = 4.0 / (self.config.get('gan_layer_count') + 1)
            scale_weights = 1.0 / self.config.get('gan_scale_count')
            for i in range(len(pred_gen_fake)):
                for j in range(len(pred_gen_fake[i]) - 1):
                    loss_gen_gan_feat += layer_weights * scale_weights * \
                        F.l1_loss(
                            pred_gen_fake[i][j],
                            pred_dis_real[i][j].detach(),
                        )

            # Generator VGG feature matching loss.
            loss_gen_vgg_feat = 0.0
            vgg_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
            out_vgg_fake = self.vgg(fake_images)
            out_vgg_real = self.vgg(real_images)
            for i in range(len(vgg_weights)):
                loss_gen_vgg_feat += vgg_weights[i] * \
                    F.l1_loss(
                        out_vgg_fake[i],
                        out_vgg_real[i],
                    )

            # Generator backpropagation.
            self.gen_optimizer.zero_grad()
            (
                loss_gen_gan +
                loss_gen_gan_feat * self.gan_layers_loss_coeff +
                loss_gen_vgg_feat * self.gan_layers_loss_coeff
            ).backward()
            self.gen_optimizer.step()

            # Discriminator backpropagation.
            self.dis_optimizer.zero_grad()
            (
                loss_dis_real +
                loss_dis_fake
            ).backward()
            self.dis_optimizer.step()

            loss_dis_fake_meter.update(loss_dis_fake.item())
            loss_dis_real_meter.update(loss_dis_real.item())
            loss_gen_gan_meter.update(loss_gen_gan.item())
            loss_gen_gan_feat_meter.update(loss_gen_gan_feat.item())
            loss_gen_vgg_feat_meter.update(loss_gen_vgg_feat.item())

            self.iter += 1

        print(
            ("TRAIN {} " + \
             "loss_dis_fake {:.5f} " + \
             "loss_dis_real {:.5f} " + \
             "loss_gen_gan {:.5f} " + \
             "loss_gen_gan_feat {:.5f} " + \
             "loss_gen_vgg_feat {:.5f}").
            format(
                self.batch_count,
                loss_dis_fake_meter.avg,
                loss_dis_real_meter.avg,
                loss_gen_gan_meter.avg,
                loss_gen_gan_feat_meter.avg,
                loss_gen_vgg_feat_meter.avg,
            ))
        sys.stdout.flush()

        if self.tb_writer is not None:
            self.tb_writer.add_scalar('train/loss/dis/fake', loss_dis_fake_meter.avg, self.batch_count)
            self.tb_writer.add_scalar('train/loss/dis/real', loss_dis_real_meter.avg, self.batch_count)
            self.tb_writer.add_scalar('train/loss/gen/gan', loss_gen_gan_meter.avg, self.batch_count)
            self.tb_writer.add_scalar('train/loss/gen/gan_feat', loss_gen_gan_feat_meter.avg, self.batch_count)
            self.tb_writer.add_scalar('train/loss/gen/vgg_feat', loss_gen_vgg_feat_meter.avg, self.batch_count)

        self.batch_count += 1

    def batch_test(self):
        self.generator.train()

        loss_gen_l1_meter = Meter()
        loss_gen_vgg_feat_meter = Meter()

        for it, (labels, real_images) in enumerate(self.test_loader):
            fake_images = self.generator(labels)

            # Generator VGG feature matching loss.
            loss_gen_vgg_feat = 0.0
            vgg_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
            out_vgg_fake = self.vgg(fake_images)
            out_vgg_real = self.vgg(real_images)
            for i in range(len(vgg_weights)):
                loss_gen_vgg_feat += vgg_weights[i] * \
                    F.l1_loss(
                        out_vgg_fake[i],
                        out_vgg_real[i],
                    )

            # Generator L1 loss.
            loss_gen_l1 = F.l1_loss(
                fake_images,
                real_images,
            )

            loss_gen_vgg_feat_meter.update(loss_gen_vgg_feat.item())
            loss_gen_l1_meter.update(loss_gen_l1.item())

            if self.tb_writer is not None:
                self.tb_writer.add_image(
                    'test/fake_images/{}'.format(it),
                    torchvision.utils.make_grid([
                        ((labels[0] + 1.0) * 127.5).cpu(),
                        ((real_images[0] + 1.0) * 127.5).cpu(),
                        ((fake_images[0] + 1.0) * 127.5).cpu(),
                    ]),
                    self.batch_count,
                )
                torchvision.utils.save_image(
                    ((fake_images[0] + 1.0) * 127.5).cpu(),
                    self.save_dir + '/test_fake_images.png',
                )


        print(
            ("TEST {} " + \
             "loss_gen_l1 {:.5f} " + \
             "loss_gen_vgg_feat {:.5f}").
            format(
                self.batch_count,
                loss_gen_l1_meter.avg,
                loss_gen_vgg_feat_meter.avg,
            ))
        sys.stdout.flush()

        if self.tb_writer is not None:
            self.tb_writer.add_scalar('test/loss/gen/l1', loss_gen_l1_meter.avg, self.batch_count)
            self.tb_writer.add_scalar('test/loss/gen/vgg_feat', loss_gen_vgg_feat_meter.avg, self.batch_count)

        if self.save_dir:
            print(
                ("Saving models and optimizers: iter={} " + \
                 "loss_gen_l1={} " + \
                 "loss_gen_vgg_feat={}").
                format(
                    self.batch_count,
                    loss_gen_l1_meter.avg,
                    loss_gen_vgg_feat_meter.avg,
                ))
            torch.save(self.generator.state_dict(), self.save_dir + "/generator.pt")
            torch.save(self.discriminator.state_dict(), self.save_dir + "/discriminator.pt")
            torch.save(self.gen_optimizer.state_dict(), self.save_dir + "/gen_optimizer.pt")
            torch.save(self.dis_optimizer.state_dict(), self.save_dir + "/dis_optimizer.pt")

