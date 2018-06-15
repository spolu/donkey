import json
import os
import cv2
import random

import numpy as np

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
    'raspi_pozyx_position',
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
    'inferred_track_progress',
    'inferred_track_position',
    'inferred_track_angle',
    'simulation_throttle',
    'simulation_steering',
    'raspi_throttle',
    'raspi_steering',
]

_target_params = [
    'corrected_track_progress',
    'corrected_track_position',
    # 'corrected_track_angle',
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
        return target_ready

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
                if data[p] is not None:
                    d[p] = data[p]
                else:
                    if p in d:
                        d.pop(p)

        if self.target_ready(index):
            t = []
            for p in _target_params:
                if p == 'corrected_track_progress':
                    t += [d[p][0]]
                    t += [d[p][1]]
                else:
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
CaptureSet is a set of captures.
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

"""
StackCaptureSet is a set of captures whose data.Dataset interface returns stack
frames with dropout (variable speeds despite fixed speed capture set).
"""
class StackCaptureSet(CaptureSet):
    def __init__(self, data_dir, stack_size=3, device=torch.device('cpu')):
        super(StackCaptureSet, self).__init__(data_dir, device=device)
        self.stack_indices = []

        for c in range(len(self.captures)):
            for d in range(1): # for now do not dropout
                for j in range(stack_size+d, len(self.captures[c].ready)):
                    indices = [k for k in range(j-(stack_size+d), j)]
                    for k in range(d):
                        indices.remove(random.choice(indices[:-1]))
                    self.stack_indices.append(
                        [[c, self.captures[c].ready[i]] for i in indices]
                    )

    # data.Dataset interface (sum of captures camera/target ready frames)

    def __getitem__(self, index):
        stack = []
        for c in self.stack_indices[index]:
            item = self.captures[c[0]].get_item(c[1])
            stack.append(item['input'])

        last = self.captures[self.stack_indices[index][-1][0]].get_item(
            self.stack_indices[index][-1][1]
        )

        return torch.cat(stack, 0), last['target']

    def __len__(self):
        return len(self.stack_indices)

"""
IMUStackCaptureSet is a set of captures whose data.Dataset interface returns
stack frames with dropout (variable speeds despite fixed speed capture set).
"""
class IMUStackCaptureSet(StackCaptureSet):
    def __init__(self, data_dir, stack_size=3, device=torch.device('cpu')):
        super(IMUStackCaptureSet, self).__init__(
            data_dir, stack_size=stack_size, device=device,
        )
    # data.Dataset interface (sum of captures camera/target ready frames)

    def __getitem__(self, index):
        stack = []
        for c in self.stack_indices[index]:
            item = self.captures[c[0]].get_item(c[1])
            stack.append(item['input'])

        last = self.captures[self.stack_indices[index][-1][0]].get_item(
            self.stack_indices[index][-1][1]
        )

        imu_values = []
        for i in range(1, len(self.stack_indices[index])):
            added = 0
            indices = self.stack_indices[index]
            for j in reversed(range(indices[i-1][1], indices[i][1])):
                if added == 3:
                    break
                item = self.captures[indices[i-1][0]].get_item(j)
                imu_values.append(torch.tensor([
                    last['time'] - item['time'],
                    item['raspi_imu_angular_velocity'][0],
                    item['raspi_imu_angular_velocity'][1],
                    item['raspi_imu_angular_velocity'][2],
                    item['raspi_imu_acceleration'][0],
                    item['raspi_imu_acceleration'][1],
                    item['raspi_imu_acceleration'][2],
                ]).to(self.device))
                added += 1

        return (torch.cat(imu_values, 0), torch.cat(stack, 0)), last['target']
