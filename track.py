import numpy as np

"""
Track interface
"""

# import pdb; pdb.set_trace()

class Track:
    def __init__(self):
        self.points = np.array(
            np.loadtxt('track.coordinates', delimiter=','),
        )

    def prev(self, point):
        if point - 1 < 0:
            return len(self.points) - 1
        else:
            return point - 1

    def next(self, point):
        if point + 1 >= len(self.points):
            return 0
        else:
            return point + 1

    def closest_pair(self, position, offset=0):
        """
        Returns the ordered closest pair of track point to the provided
        position, properly wraping around. It's a bit involved but should work
        always.
        """
        deltas = self.points - position
        sq_distances = np.sum(deltas**2, axis=1)
        closest = np.argmin(sq_distances)

        for i in range(offset):
            closest = self.next(closest)

        candidates = []
        distances = []

        candidates.append(self.prev(closest))
        distances.append(sq_distances[self.prev(closest)])
        candidates.append(self.next(closest))
        distances.append(sq_distances[self.next(closest)])

        if distances[1] > distances[0]:
            return [
                candidates[0],
                closest,
            ]
        else:
            return [
                closest,
                candidates[1],
            ]

    def unity(self, position, offset=0):
        """
        Returns the unity vector of the track at the closest point from the
        given position.
        """
        closests = self.closest_pair(position, offset)
        u = self.points[closests[0]]
        v = self.points[closests[1]]
        t = (v-u) / np.linalg.norm(v-u)
        return t

    def position(self, position, offset=0):
        """
        Returns the position of the car on the track axis orthogonal.
        """
        closests = self.closest_pair(position, offset)
        u = self.points[closests[0]] - position
        v = self.points[closests[1]] - position
        return (np.cross(u, v) / np.linalg.norm(u-v))[1]

    def angle(self, position, velocity, offset=0):
        t = self.unity(position, offset)
        w = velocity / np.linalg.norm(velocity)
        return np.sign(np.linalg.det([
            t,
            w,
            [0,1,0],
        ])) * np.arccos(np.dot(t, w))

    def divergence(self, position, offset=0):
        """
        Returns the distance of the position to the track.
        """
        closests = self.closest_pair(position, offset)
        u = self.points[closests[0]] - position
        v = self.points[closests[1]] - position
        return np.linalg.norm(np.cross(u, v)) / np.linalg.norm(u-v)

    def speed(self, position, velocity, offset=0):
        """
        Returns the linear speed along the track given a position and a
        velocity.
        """
        t = self.unity(position, offset)
        return np.dot(t, velocity)

    def lateral_speed(self, position, velocity, offset=0):
        t = self.unity(position, offset)
        return np.cross(t, velocity)[1]
