import json
import os

import torch.utils.data as data

"""
Capture interface
"""
class Capture(data.Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.data_dir = data_dir

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

            self.data.append({
                'track_angles': data['track_angles'],
                'track_position': data['track_position'],
                'camera': camera,
            })

            index += 1

    def __saveitem__(self, index):
        assert self.data[index] is not None

        with open(os.path.join(self.data_dir, str(index) + '.json'), "w+") as f:
            json.dump({
                'track_angles': self.data[index]['track_angles'],
                'track_position': self.data[index]['track_position'],
            }, f)
        with open(os.path.join(self.data_dir, str(index) + '.jpeg'), "wb+") as f:
            f.write(self.data[index]['camera'])

    def __additem__(self, camera, track_angles, track_position):
        index = len(self.data)
        self.data.append({
            'camera': camera,
            'track_angles': track_angles,
            'track_position': track_position,
        })
        self.__saveitem__(index)

    def __getitem__(self, index):
        assert index < len(self.data)
        item = self.data[index]
        return item['camera'], item['track_angles'], item['track_position']

    def __len__(self):
        return len(self.data)

