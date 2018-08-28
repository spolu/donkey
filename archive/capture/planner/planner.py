import numpy as np
import torch

from track import Track

MAVG_LEN = 10

class Planner:
    def __init__(self, config):
        self.track_name = config.get('track_name')

        self.track = Track(self.track_name)
        self.last_position = None

        self.last_track_progress = [0.0] * MAVG_LEN
        self.last_track_position = [0.0] * MAVG_LEN
        # self.last_track_angle = [0.0] * MAVG_LEN

        self.last_position = np.array([0.0, 0.0, 0.0])

    def plan(self, track_coordinates):
        track_progress = self.track.progress(track_coordinates)
        track_position = self.track.position(track_coordinates)

        self.last_track_progress = self.last_track_progress[1:] + [track_progress]
        self.last_track_position = self.last_track_position[1:] + [track_position]
        # self.last_track_angle = self.last_track_angle[1:] + [track_angle]

        # track_progress = np.mean(self.last_track_progress)
        # track_position = np.mean(self.last_track_position)
        # track_angle = np.mean(self.last_track_angle)

        steering = 0.0

        # position = self.track.invert(track_coordinates)
        # unity = self.track.unity(position)
        # future_angle = self.track.angle(position, unity, 4) + track_angle

        print(">>>>>>>>>>>>>>> TRACK_PROGRESS: {}".format(track_progress))
        print(">>>>>>>>>>>>>>> TRACK_POSITION: {}".format(track_position))
        # print(">>>>>>>>>>>>>>> TRACK_ANGLE: {}".format(track_angle))
        # print(">>>>>>>>>>>>>>> POSITION: {}".format(position))
        # print(">>>>>>>>>>>>>>> UNITY: {}".format(unity))
        # print(">>>>>>>>>>>>>>> FUTURE ANGLE: {}".format(future_angle))

        # if future_angle > 0.25:
        #     print("################### FUTURE_ANGLE STEERING")
        #     steering = -min(1.0, 2 * future_angle)
        # elif future_angle < -0.25:
        #     print("################### FUTURE_ANGLE STEERING")
        #     steering = min(1.0, -2 * future_angle)

        if track_position > 0:
            steering = -min(1.0, 5 * track_position)
        if track_position < 0:
            steering = min(1.0, 5 * -track_position)

        print("COMMAND: {}".format(steering))

        return steering, 0.6
