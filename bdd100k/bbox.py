import json
import os
import cv2
import random
import sqlite3

import numpy as np

import torch
import torch.utils.data as data

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.path import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg

_labels = [
    'traffic sign', 'car', 'truck', 'bus', 'caravan', 'motorcycle', 'trailer',
    'motor', 'lane/road curb', 'lane/single white', 'lane/single yellow',
]

class BDD100kBBox(data.Dataset):
    def __init__(self, config, validation=False):
        super(BDD100kBBox, self).__init__()
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
        return len(_labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]

        raw_image = cv2.imread(
            os.path.join(
                self.data_dir, 'images/100k', self.dataset, name + '.jpg'
            ),
        )
        image = cv2.resize(
            raw_image,
            (1024, 576),
            interpolation = cv2.INTER_CUBIC,
        )[32:544, :] / 127.5 - 1.0

        raw_labels = np.zeros((720, 1280, len(_labels)))

        objects = []
        with open(os.path.join(self.data_dir, 'labels/100k/' + self.dataset + '/' + name + '.json'), "r") as f:
            label = json.load(f)
            for o in label['frames'][0]['objects']:
                # print("{}".format(o))
                if o['category'] in _labels and (
                        'direction' not in o['attributes'] or \
                        o['attributes']['direction'] == 'parallel'
                ):
                    objects.append(o)

        for i, l in enumerate(_labels):
            fig = plt.figure(figsize=(16, 9), dpi=80)
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
            ax.set_xlim(0, 1280 - 1)
            ax.set_ylim(0, 720 - 1)

            # ax.imshow(raw_image, interpolation='nearest', aspect='auto')

            for o in objects:
                if o['category'] == l and 'box2d' in o:
                    x1 = o['box2d']['x1']
                    y1 = o['box2d']['y1']
                    x2 = o['box2d']['x2']
                    y2 = o['box2d']['y2']

                    rect = mpatches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=1.0,
                        edgecolor=[0, 0, 0],
                        facecolor=[0, 0, 0],
                        fill=True, alpha=1.0,
                    )

                    ax.add_patch(rect)
                    # print("ADDED BOX2D {} {} {} {}".format(x1, x2, x2, y2))

                if o['category'] == l and 'poly2d' in o:
                    moves = {'L': Path.LINETO,
                             'C': Path.CURVE4}
                    points = [p[:2] for p in o['poly2d']]
                    codes = [moves[p[2]] for p in o['poly2d']]
                    codes[0] = Path.MOVETO

                    poly = mpatches.PathPatch(
                        Path(points, codes),
                        linewidth=1.0,
                        edgecolor=[0, 0, 0],
                        facecolor='none',
                        antialiased=False, alpha=1.0, snap=True,
                    )

                    ax.add_patch(poly)
                    # print("ADDED POLY2D")

            ax.invert_yaxis()

            canvas = plt.get_current_fig_manager().canvas
            agg = canvas.switch_backends(FigureCanvasAgg)
            agg.draw()

            data = np.fromstring(agg.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(agg.get_width_height()[::-1] + (3,))

            # cv2.imwrite("test_{}.png".format(i), data)
            raw_labels[:,:,i:i+1] = np.copy(data[:,:,0:1])

        labels = cv2.resize(
            raw_labels,
            (1024, 576),
            interpolation = cv2.INTER_CUBIC,
        )[32:544, :] / 127.5 - 1.0

        image = torch.from_numpy(image).float().transpose(2, 0).transpose(1, 2).to(self.device)
        labels = torch.from_numpy(labels).float().transpose(2, 0).transpose(1, 2).to(self.device)
        # import pdb; pdb.set_trace()

        return (labels, image)

