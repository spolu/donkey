import numpy as np

"""
Track interface
"""

# import pdb; pdb.set_trace()

class Track:
    def __init__(self, points):
        self.points = points

    def closest_point(self, point):
        deltas = self.points - point
        sq_distances = np.sum(deltas**2, axis=1)
        closest = np.argmin(sq_distances)
        return self.points[closest]

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
        closests = self.closest_pair(position)
        u = self.points[closests[0]] - position
        v = self.points[closests[1]] - position
        return np.linalg.norm(np.cross(u, v)) / np.linalg.norm(u - v)

    def speed(self, position, velocity):
        closests = self.closest_pair(position)
        u = self.points[closests[0]]
        v = self.points[closests[1]]
        t = (v - u) / np.linalg.norm(v - u)
        return np.dot(t, velocity)

    def finish_line(self, prev_position, next_position):
        pass

    def scalar_to_direction(self, point):
        deltas = self.points - point
        sq_distances = np.sum(deltas**2, axis=1)
        minIndex = np.argmin(sq_distances)

        #TODO doesn't handle well the last iteam of the array
        tangent = self.points[minIndex+1] - self.points[minIndex]
        normalized_tangent = tangent/np.sqrt(np.sum(tangent**2))
        vector_from_track = point - self.points[minIndex]
        scalar = np.dot(tangent,vector_from_track)
        return scalar

def main():
    # loading the track points
    trackPoints = np.loadtxt('track_coordinates.txt',delimiter=',')
    track = Track(np.array(trackPoints))

    distance = track.distance([60.35,0.5,35.6])
    speed = track.speed([61.0,0.0,41.0], [0.0, 0.0, 4.0])

    track.closest_pair([61.0,0.0,41.0])

    print(distance)
    print(speed)

if __name__ == "__main__":
    main()
