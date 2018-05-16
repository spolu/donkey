import numpy as np
import torch

class Planner:
    def __init__(self):
        pass

    def plan(self, track_progress, track_position, track_angle):
        steering = 0.0
        throttle_brake = 0.5

        if track_position > 0 and track_angle > 0:
            # Immediate correction.
            immediate = 5 * max(track_angle.item(), track_position.item())
            # Cap.
            cap = min(1.0, immediate)

            steering = -cap

        if track_position < 0 and track_angle < 0:
            # Immediate correction.
            immediate = 5 * max(-track_angle.item(), -track_position.item())
            # Cap.
            cap = min(1.0, immediate)

            steering = cap

        print("COMMAND: {}".format(steering))

        return steering, throttle_brake
