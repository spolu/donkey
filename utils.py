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

class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X
