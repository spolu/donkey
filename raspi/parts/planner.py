import time
from planner import Planner as Planr

class Planner:
    def __init__(self, config):
        self.planner = Planr(config)

    def run(self, track_coordinates):
        return self.planner.plan(track_coordinates)

