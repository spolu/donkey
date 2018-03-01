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

    def closest_pair(self, position):
        """
        Returns the ordered closest pair of track point to the provided
        position, properly wraping around. It's a bit involved but should work
        always.
        """
        deltas = self.points - position
        sq_distances = np.sum(deltas**2, axis=1)
        closest = np.argmin(sq_distances)

        candidates = []
        distances = []

        if closest-1 < 0:
            candidates.append(len(self.points)-1)
            distances.append(sq_distances[-1])
        else:
            candidates.append(closest-1)
            distances.append(sq_distances[closest-1])
        if closest+1 >= len(self.points):
            candidates.append(0)
            distances.append(sq_distances[0])
        else:
            candidates.append(closest+1)
            distances.append(sq_distances[closest+1])

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

    def distance(self, position):
        """
        Returns the distance of the position to the track.
        """
        closests = self.closest_pair(position)
        u = self.points[closests[0]] - position
        v = self.points[closests[1]] - position
        return np.linalg.norm(np.cross(u, v)) / np.linalg.norm(u-v)

    def unity(self, position):
        """
        Returns the unity vector of the track at the closest point from the
        given position.
        """
        closests = self.closest_pair(position)
        u = self.points[closests[0]]
        v = self.points[closests[1]]
        t = (v-u) / np.linalg.norm(v-u)
        return t


    def speed(self, position, velocity):
        """
        Returns the linear speed along the track given a position and a
        velocity.
        """
        closests = self.closest_pair(position)
        u = self.points[closests[0]]
        v = self.points[closests[1]]
        t = (v-u) / np.linalg.norm(v-u)
        return np.dot(t, velocity)

    # TODO(stan): this won't work until we prevent a,b from being returned in
    #             the following situation (where b,c should be returned):
    #             (a)-(b)--(p)-------(c)
    # def finish_line(self, prev_position, next_position):
    #     prev_closests = self.closest_pair(prev_position)
    #     next_closests = self.closest_pair(next_position)
    #     return prev_closests[1] == 0 and next_closests[0] == 0

# def main():
#     # loading the track points
#     track = Track()
#
#     distance = track.distance([60.35,0.5,35.6])
#     speed = track.speed([61.0,0.0,41.0], [0.0, 0.0, 4.0])
#     finish_line = track.finish_line([46.8,0.633,50.1], [46.9,0.633,53.0])
#     speed = track.speed([46.8,0.633,50.1], [0.0, 0.0, 2.0])
#
#
#     print(distance)
#     print(speed)
#     print(finish_line)
#
# if __name__ == "__main__":
#     main()
