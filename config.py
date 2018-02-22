import json
import os

class Config:
    def __init__(self, path):
        self.config = json.load(open(os.path.abspath(path), 'r'))

    def get(self, key):
        if key not in self.config:
            raise Exception("Unknown Config key {}".format(key))
        return self.config[key]

    def __eq__(self, other):
        """Overrides the default implementation"""
        a, b = json.dumps(self.config, sort_keys=True), json.dumps(other.config, sort_keys=True)
        return a == b

