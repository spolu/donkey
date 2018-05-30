import json
import os
import numpy as np
import cv2

import torch
import torch.utils.data as data

_stored_params = [
    'time',
    'simulation_angular_velocity',
    'simulation_acceleration',
    'raspi_imu_angular_velocity',
    'raspi_imu_acceleration',
    'raspi_sensehat_angular_velocity',
    'raspi_sensehat_acceleration',
    'raspi_phone_position',
    'annotated_track_progress',
    'annotated_track_position',
    'annotated_track_angle',
    'reference_track_progress',
    'reference_track_position',
    'reference_track_angle',
    'integrated_track_progress',
    'integrated_track_position',
    'integrated_track_angle',
    'corrected_track_progress',
    'corrected_track_position',
    'corrected_track_angle',
    'simulation_throttle',
    'simulation_steering',
    'raspi_throttle',
    'raspi_steering',
]

_target_params = [
    'corrected_track_progress',
    'corrected_track_position',
    'corrected_track_angle',
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
    def __init__(self, data_dir, load=True, device=torch.device('cpu')):
        self.data = []
        self.data_dir = data_dir
        self.device = device

        self.sequence_cache = None

        first = True
        found = True
        index = 0

        while found and load:
            if not os.path.isfile(os.path.join(self.data_dir, str(index) + '.json')):
                found = False
            if not os.path.isfile(os.path.join(self.data_dir, str(index) + '.jpeg')):
                found = False

            if not found and not first:
                continue
            if not found and first:
                found = True
                index += 1
                continue

            first = False

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

        target_ready = True
        for p in _target_params:
            if p not in d:
                target_ready = False

        if target_ready:
            t = []
            for p in _target_params:
                t += [d[p]]
            d['target'] = torch.tensor(t, dtype=torch.float).to(self.device)

        if save:
            self.save_item(index)

    def get_item(self, index):
        assert index < len(self.data)
        return self.data[index]

    def size(self):
        return len(self.data)

    def sequence(self):
        if self.sequence_cache is None:
            sequence = []
            for i in range(self.size()):
                item = self.get_item(i)
                sequence.append(item['input'])

            self.sequence_cache = torch.stack(sequence).to(self.device)

        return self.sequence_cache

    # data.Dataset interface.

    def __getitem__(self, index):
        item = self.get_item(index)
        return item['input'], item['target']

    def __len__(self):
        return self.size()

"""
A CaptureSet is a set of captures.
"""
class CaptureSet(data.Dataset):
    def __init__(self, data_dir, device=torch.device('cpu')):
        self.captures = []
        self.data_dir = data_dir
        self.device = device

        self.dataset_len = 0
        for d in [os.path.join(self.data_dir, s) for s in next(os.walk(self.data_dir))[1]]:
            c = Capture(d, load=True, device=self.device)
            self.captures.append(c)
            self.dataset_len += c.__len__()

    def size(self):
        return len(self.captures)

    def get_capture(self, index):
        assert index < len(self.captures)
        return self.captures[index]
