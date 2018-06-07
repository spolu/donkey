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
    'raspi_sensehat_orientation',
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
        self.offset = 0
        self.ready = []

        self.sequence_cache = None

        first = True
        found = True
        index = 0

        while found and load:
            if not os.path.isfile(os.path.join(self.data_dir, str(index) + '.json')):
                found = False

            if not found and not first:
                continue
            if not found and first:
                found = True
                index += 1
                continue

            if first:
                self.offset = index
            first = False

            data = None
            camera = None

            with open(os.path.join(self.data_dir, str(index) + '.json'), "r") as f:
                data = json.load(f)
            if os.path.isfile(os.path.join(self.data_dir, str(index) + '.jpeg')):
                with open(os.path.join(self.data_dir, str(index) + '.jpeg'), "rb") as f:
                    camera = f.read()

            self.add_item(
                camera,
                data,
                save=False,
            )

            index += 1

        print("Capture loaded: offset={} size={} data_dir={}".format(
            self.offset, len(self.data), data_dir,
        ))

    def save_item(self, index):
        assert self.data[index] is not None

        with open(os.path.join(self.data_dir, str(self.offset + index) + '.json'), "w+") as f:
            d = {}
            for p in _stored_params:
                if p in self.data[index]:
                    d[p] = self.data[index][p]
            json.dump(d, f)

        if 'camera' in self.data[index]:
            with open(os.path.join(self.data_dir, str(self.offset + index) + '.jpeg'), "wb+") as f:
                f.write(self.data[index]['camera'])

        # print("CAPTURE: save_item {}".format(self.offset + index))

    def target_ready(self, index):
        d = self.data[index]

        target_ready = True
        for p in _target_params:
            if p not in d:
                target_ready = False

    def add_item(self, camera, data, save=True):
        index = len(self.data)

        if camera is not None:
            self.data.append({
                'camera': camera,
                'input': input_from_camera(camera, self.device),
            })
        else:
            self.data.append({})

        # print("CAPTURE: add_item {}".format(index))

        self.update_item(index, data, save=save)

        if camera is not None and self.target_ready(index):
            self.ready.append(index)

    def update_item(self, index, data, save=True):
        assert index < len(self.data)

        d = self.data[index]
        for p in _stored_params:
            if p in data:
                d[p] = data[p]

        if self.target_ready(index):
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

    def save(self):
        for i in range(len(self.data)):
            self.update_item(i, {}, save=True)

    # data.Dataset interface (only camera/target ready frames)

    def __getitem__(self, index):
        assert index < len(self.ready)
        item = self.get_item(self.ready[index])
        return item['input'], item['target']

    def __len__(self):
        return len(self.ready)

"""
A CaptureSet is a set of captures.
"""
class CaptureSet(data.Dataset):
    def __init__(self, data_dir, device=torch.device('cpu')):
        self.captures = []
        self.data_dir = data_dir
        self.device = device

        for d in [os.path.join(self.data_dir, s) for s in next(os.walk(self.data_dir))[1]]:
            c = Capture(d, load=True, device=self.device)
            self.captures.append(c)

    def size(self):
        return len(self.captures)

    def get_capture(self, index):
        assert index < len(self.captures)
        return self.captures[index]

    # data.Dataset interface (sum of captures camera/target ready frames)

    def __getitem__(self, index):
        for i in range(len(self.captures)):
            if index >= len(self.captures[i].ready):
                index -= len(self.captures[i].ready)
                continue
            else:
                assert index < len(self.captures[i].ready)
                item = self.captures[i].get_item(self.captures[i].ready[index])
                return item['input'], item['target']

    def __len__(self):
        length = 0
        for i in range(len(self.captures)):
            length += len(self.captures[i].ready)
        return length
