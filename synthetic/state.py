import numpy as np

class State:
    def __init__(
            self,
            track_randomization,
            position,
            velocity,
            angular_velocity,
            track_coordinates,
            track_angle,
    ):
        self.track_randomization = track_randomization
        self.position = position
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.track_coordinates = track_coordinates
        self.track_angle = track_angle

        self._vector = np.array([
            # track_randomization,
            track_coordinates[0],
            track_coordinates[1],
            track_coordinates[2],
            track_angle,
            # self.velocity[0],
            # self.velocity[1],
            # self.velocity[2],
            # self.angular_velocity[0],
            # self.angular_velocity[1],
            # self.angular_velocity[2],
        ])

    def vector(self):
        return self._vector

    @staticmethod
    def size():
        return 4
