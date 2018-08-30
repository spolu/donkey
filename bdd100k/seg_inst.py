import json
import os
import cv2
import random
import sqlite3

import numpy as np

import torch
import torch.utils.data as data

class BDD100kSegInst(data.Dataset):
    def __init__(self, config, validation=False):
        super(BDD100kSegInst, self).__init__()
        self.device = torch.device(config.get('device'))
        self.config = config

        self.data_dir = config.get('bdd100k_data_dir')
        self.sqlite_path = config.get('bdd100k_sqlite_path')

        assert self.data_dir is not None
        assert self.sqlite_path is not None

        self._conn = sqlite3.connect(self.sqlite_path)
        self.validation = validation

        self.dataset = 'val' if self.validation else 'train'

        self.images = []
        c = self._conn.cursor()
        for r in c.execute('''
SELECT name FROM labels
WHERE scene='highway'
  AND segmented='true'
  AND active='true'
  AND dataset=?
        ''', [self.dataset]):
            self.images.append(r[0])

    def output_channel_count(self):
        return 3

    def input_channel_count(self):
        return 3

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]

        image = cv2.resize(
            cv2.imread(
                os.path.join(
                    self.data_dir, 'images/100k', self.dataset, name + '.jpg'
                ),
            ),
            (1024, 576),
            interpolation = cv2.INTER_CUBIC,
        )[32:544, :] / 127.5 - 1.0

        labels = cv2.resize(
            cv2.imread(
                os.path.join(
                    self.data_dir, 'seg/color_labels', self.dataset, name + '_train_color.png'
                ),
            ),
            (1024, 576),
            interpolation = cv2.INTER_NEAREST,
        )[32:544, :] / 127.5 - 1.0

        image = torch.from_numpy(image).float().transpose(2, 0).to(self.device)
        labels = torch.from_numpy(labels).float().transpose(2, 0).to(self.device)

        return (labels, image)

