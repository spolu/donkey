import json
import os
import cv2
import random

import numpy as np

import torch
import torch.utils.data as data

_stored_params = [
    'time',
    'simulation_track_randomization',
    'simulation_position',
    'simulation_velocity',
    'simulation_angular_velocity',
    'simulation_track_coordinates',
    'simulation_track_angle',
    'simulation_steering',
    'simulation_throttle',
    'simulation_brake',
    'raspi_throttle',
    'raspi_steering',
    'raspi_imu_angular_velocity',
    'raspi_imu_acceleration',
    'raspi_sensehat_orientation',
    'raspi_phone_position',
    'raspi_pozyx_position',
]

"""
Capture interface
"""
class Capture(data.Dataset):
    def __init__(self, data_dir, load=True, loader=lambda item: item):
        self.data = []
        self.data_dir = data_dir
        self.loader=loader
        self.offset = 0
        self.ready = []

        self.sequence_cache = None

        first = True
        found = True
        index = 0

        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

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

    def add_item(self, camera, data, save=True):
        index = len(self.data)

        if camera is not None:
            self.data.append({
                'camera': camera,
            })
        else:
            self.data.append({})

        # print("CAPTURE: add_item {}".format(index))

        self.update_item(index, data, save=save)

        if camera is not None:
            self.ready.append(index)

    def update_item(self, index, data, save=True):
        assert index < len(self.data)

        d = self.data[index]
        for p in _stored_params:
            if p in data:
                if data[p] is not None:
                    d[p] = data[p]
                else:
                    if p in d:
                        d.pop(p)

        if save:
            self.save_item(index)

    def get_item(self, index):
        assert index < len(self.data)
        return self.data[index]

    def size(self):
        return len(self.data)

    def save(self):
        for i in range(len(self.data)):
            self.update_item(i, {}, save=True)

    # data.Dataset interface (only camera/target ready frames)

    def __getitem__(self, index):
        assert index < len(self.ready)
        item = self.get_item(self.ready[index])
        return self.loader(item)

    def __len__(self):
        return len(self.ready)

"""
CaptureSet is a set of captures.
"""
class CaptureSet(data.Dataset):
    def __init__(self, data_dir, loader=lambda item: item):
        self.captures = []
        self.data_dir = data_dir
        self.loader = loader

        for d in [os.path.join(self.data_dir, s) for s in next(os.walk(self.data_dir))[1]]:
            c = Capture(d, load=True)
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
                return self.loader(item)

    def __len__(self):
        length = 0
        for i in range(len(self.captures)):
            length += len(self.captures[i].ready)
        return length

