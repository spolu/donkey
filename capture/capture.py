import json
import os
import numpy as np
import cv2

import torch
import torch.utils.data as data

_stored_params = [
    'angular_velocity',
    'reference_progress',
    'reference_track_position',
    'reference_track_angle',
    'progress',
    'track_position',
    'track_angle',
]

def input_from_camera(camera, device):
    tensor = torch.tensor(cv2.imdecode(
        np.fromstring(camera, np.uint8),
        cv2.IMREAD_COLOR,
    ), dtype=torch.float).to(device)
    tensor = tensor / 127.5 - 1
    tensor = tensor.transpose(0, 2)

    return tensor

"""
Capture interface
"""
class Capture(data.Dataset):
    def __init__(self, data_dir, device=torch.device('cpu')):
        self.data = []
        self.data_dir = data_dir
        self.device = device

        found = True
        index = 0
        while found:
            if not os.path.isfile(os.path.join(self.data_dir, str(index) + '.json')):
                found = False
            if not os.path.isfile(os.path.join(self.data_dir, str(index) + '.jpeg')):
                found = False
            if not found:
                continue

            data = None
            camera = None
            with open(os.path.join(self.data_dir, str(index) + '.json'), "r") as f:
                data = json.load(f)
            with open(os.path.join(self.data_dir, str(index) + '.jpeg'), "rb") as f:
                camera = f.read()

            self.add_item(
                camera,
                data,
                save=False,
            )

            index += 1

    def save_item(self, index):
        assert self.data[index] is not None

        with open(os.path.join(self.data_dir, str(index) + '.json'), "w+") as f:
            d = {}
            for p in _stored_params:
                if p in self.data[index]:
                    d[p] = self.data[index][p]
            json.dump(d, f)
        with open(os.path.join(self.data_dir, str(index) + '.jpeg'), "wb+") as f:
            f.write(self.data[index]['camera'])

    def add_item(self, camera, data, save=True):
        index = len(self.data)

        self.data.append({
            'camera': camera,
            'input': input_from_camera(camera, self.device),
        })
        self.update_item(index, data, save=save)

    def update_item(self, index, data, save=True):
        assert index < len(self.data)

        d = self.data[index]
        for p in _stored_params:
            if p in data:
                d[p] = data[p]

        if d['progress'] and d['track_position'] and d['track_angle']:
            target = torch.tensor(
                [d['progress']] + [d['track_position']] + [d['track_angle']],
                dtype=torch.float,
            ).to(self.device)
        else:
            target = None
        if save:
            self.save_item(index)

    def get_item(self, index):
        assert index < len(self.data)

        item = self.data[index]
        return item['input'], item['target']

    def __getitem__(self, index):
        self.get_item(index)

    def __len__(self):
        return len(self.data)

