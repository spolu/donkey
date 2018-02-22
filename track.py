import numpy as np
from io import StringIO

"""
Track interface
"""

class Track:
    def __init__(self, points):
	    self.points = points

    def closest_point(self, point):
    	deltas = self.points - point
    	sq_distances = np.sum(deltas**2, axis=1)
    	minIndex = np.argmin(sq_distances)
        return self.points[minIndex]

    def distance_to_closest(self, point):
    	deltas = self.points - point
    	sq_distances = np.sum(deltas**2, axis=1)
    	distance_to_closest = np.amin(sq_distances)
        return np.sqrt(distance_to_closest)

    def scalar_to_direction(self, point):
    	deltas = self.points - point
    	sq_distances = np.sum(deltas**2, axis=1)
    	minIndex = np.argmin(sq_distances)
    	tangent = self.points[minIndex+1] - self.points[minIndex]
        normalized_tangent = tangent/np.sqrt(np.sum(tangent**2))
    	vector_from_track = point - self.points[minIndex]
    	scalar = np.dot(tangent,vector_from_track)
        return scalar

# loading the track points
trackPoints = np.loadtxt('sim/Assets/Resources/warehouse_path.txt',delimiter=',')
track = Track(np.array(trackPoints))

point = track.closest_point([61.0,0.0,41.0])
distance = track.distance_to_closest([61.0,0.0,41.0])
scalar = track.scalar_to_direction([61.0,0.0,41.0])