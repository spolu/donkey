import json
import os
import numpy as np
import cv2

import torch
import torch.utils.data as data

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

            self.__additem__(
                camera,
                data['track_progress'],
                data['track_position'],
                save=False,
            )

            index += 1

    def __saveitem__(self, index):
        assert self.data[index] is not None

        with open(os.path.join(self.data_dir, str(index) + '.json'), "w+") as f:
            json.dump({
                'track_progress': self.data[index]['track_progress'],
                'track_position': self.data[index]['track_position'],
            }, f)
        with open(os.path.join(self.data_dir, str(index) + '.jpeg'), "wb+") as f:
            f.write(self.data[index]['camera'])

    def __additem__(self, camera, track_progress, track_position, save=True):
        index = len(self.data)

        target = torch.tensor(
            [track_progress] + [track_position],
            dtype=torch.float,
        ).to(self.device)

        self.data.append({
            'camera': camera,
            'track_progress': track_progress,
            'track_position': track_position,
            'input': input_from_camera(camera, self.device),
            'target': target,
        })
        if save:
            self.__saveitem__(index)

    def __getitem__(self, index):
        assert index < len(self.data)
        item = self.data[index]
        return item['input'], item['target']

    def __len__(self):
        return len(self.data)

