import numpy as np

class State:
    def __init__(
            self,
            track_rotation_randomization,
            position,
            velocity,
            angular_velocity,
            track_coordinates,
            track_angle,
    ):
        self.track_rotation_randomization = track_rotation_randomization
        self.position = position
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.track_coordinates = track_coordinates
        self.track_angle = track_angle

        self._vector = np.array([
            # track_rotation_randomization,
            track_coordinates[0],
            track_coordinates[1],
            track_coordinates[2],
            track_angle,
            # np.linalg.norm(self.velocity),
            # np.linalg.norm(self.angular_velocity),
        ])

    def vector(self):
        return self._vector

    @staticmethod
    def size():
        return 4
