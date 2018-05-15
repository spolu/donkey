import json
import os
import numpy as np

class Config:
    def __init__(self, path):
        self.config = json.load(open(os.path.abspath(path), 'r'))

    def get(self, key):
        if key not in self.config:
            raise Exception("Unknown Config key {}".format(key))
        return self.config[key]

    def override(self, key, value):
        self.config[key] = value

    def __eq__(self, other):
        """Overrides the default implementation"""
        a = json.dumps(self.config, sort_keys=True)
        b = json.dumps(other.config, sort_keys=True)
        return a == b

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Meter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.max = None
        self.min = None
        self.avg = None
        self.sum = 0
        self.cnt = 0

    def update(self, val):
        self.sum += val
        self.cnt += 1
        if self.max is None or self.max < val:
            self.max = val
        if self.min is None or self.min > val:
            self.min = val
        self.avg = self.sum / self.cnt


