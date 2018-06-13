import time
from planner import Planner as Planr

class Planner:
    def __init__(self, config):
        self.planner = Planr(config)

    def run(self, track_progress, track_position, track_angle):
        return self.planner.plan(
            track_progress, track_position, track_angle,
        )

