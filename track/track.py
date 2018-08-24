import random
import os
import math

import numpy as np
from enum import Enum

"""
Track interface
"""

# import pdb; pdb.set_trace()

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)

    return np.array((
        (c, 0, s),
        (0, 1, 0),
        (-s, 0, c),
    ))

class Track:
    def __init__(self, name):
        self.name = name
        self.script = Script(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'coordinates/' + name + '.script',
        ))

        self.points = self.script.points()
        # `track_width` is used as half width in the unity script.
        self.track_width = self.script.width
        self.track_span = self.script.span

        # Iinterpolate to first point.
        self.points.append(np.array([
            0.001 * self.points[-1][0] + 0.999 * self.points[0][0],
            0.001 * self.points[-1][1] + 0.999 * self.points[0][1],
            0.001 * self.points[-1][2] + 0.999 * self.points[0][2],
        ]))

        # Recompute length.
        self.track_length = 0.0
        for i in range(len(self.points)):
            self.track_length += np.linalg.norm(
                self.points[self.next(i)] - self.points[i],
            )
        self.randomization = 0

        # Center the track in space.
        min_x = 0;
        max_x = 0;
        min_z = 0;
        max_z = 0;
        for p in self.points:
            min_x = min(min_x, p[0])
            max_x = max(max_x, p[0])
            min_z = min(min_z, p[2])
            max_z = max(max_z, p[2])
        for i in range(len(self.points)):
            self.points[i] = self.points[i] + np.array(
                [-(max_x+min_x)/2.0, 0, -(max_z+min_z)/2.0]
            )

        self.start_angle = 0.0
        self.start_position = self.points[0]
        self.rotation_randomization = 0.0
        self.start_position_randomization = 0.0
        self.initial_points = np.copy(self.points)

    def width(self):
        return self.track_width

    def span(self):
        return self.track_span

    def length(self):
        return self.track_length

    def randomize(self, start_position=True, rotation=True):
        """
        Randomize track by randomly applying symetry to the track and picking
        translating such that the starting location is picked randomly.
        """
        self.points = np.copy(self.initial_points)
        self.start_angle = 0.0
        self.start_position = self.points[0]

        if rotation:
            self.rotation_randomization = random.uniform(0, 1)
            theta = self.rotation_randomization * 2 * math.pi
            for j in range(len(self.points)):
                self.points[j] = np.dot(rot_y(theta), self.points[j])
            self.start_angle = theta
            self.start_position = self.points[0]
        else:
            self.rotation_randomization = 0.0

        if start_position:
            self.start_position_randomization = random.uniform(0, 1)
            start_index = int(
                self.start_position_randomization * len(self.points)
            ) % len(self.points)

            u = self.points[1]-self.points[0]
            u = u / np.linalg.norm(u)
            for i in range(1, start_index+1):
                v = self.points[i] - self.points[i-1]
                w = v / np.linalg.norm(v)
                theta = -np.sign(np.linalg.det([
                    u,
                    w,
                    [0,1,0],
                ])) * np.arccos(min(1.0, np.dot(u, w)))
                self.start_angle -= theta
                u = w
            self.start_position = self.points[start_index]
        else:
            self.start_position_randomization = 0.0

        # if symmetry:
        #     if random.randint(0, 1) == 1:
        #         for i in range(len(self.points)):
        #             self.points[i][0] = -self.points[i][0]
        #     if random.randint(0, 1) == 1:
        #         for i in range(len(self.points)):
        #             self.points[i][2] = -self.points[i][2]

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

    def closests(self, position, offset=0):
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
        closests = self.closests(position, offset)
        u = self.points[closests[0]]
        v = self.points[closests[1]]
        t = (v-u) / np.linalg.norm(v-u)
        return t

    def angle(self, position, velocity, offset=0):
        t = self.unity(position, offset)
        w = velocity / np.linalg.norm(velocity)
        return np.sign(np.linalg.det([
            t,
            w,
            [0,1,0],
        ])) * np.arccos(np.dot(t, w)) / math.pi

    def divergence(self, position, offset=0):
        """
        Returns the distance of the position to the track.
        """
        closests = self.closests(position, offset)
        u = self.points[closests[0]] - position
        v = self.points[closests[1]] - position
        return np.linalg.norm(np.cross(u, v)) / np.linalg.norm(u-v)

    def linear_speed(self, position, velocity, offset=0):
        """
        Returns the linear speed along the track given a position and a
        velocity.
        """
        t = self.unity(position, offset)
        return np.dot(t, velocity)

    def lateral_speed(self, position, velocity, offset=0):
        t = self.unity(position, offset)
        return np.cross(t, velocity)[1]

    def coordinates(self, position):
        closests = self.closests(position, 0)

        p = 0.0
        for i in range(closests[1]):
            p += np.linalg.norm(
                self.points[self.next(i)] - self.points[i],
            )
        t = self.unity(position, 0)
        u = position - self.points[closests[0]]
        v = self.points[closests[1]] - self.points[closests[0]]
        p -= np.linalg.norm(v) - np.dot(t, u)

        u = self.points[closests[0]] - position
        v = self.points[closests[1]] - position
        l = (np.cross(u, v) / np.linalg.norm(u-v))[1]

        return np.array([
            np.cos(2*math.pi*(p / self.track_length)),
            np.sin(2*math.pi*(p / self.track_length)),
            l / self.track_width,
        ])

    def position(self, coordinates):
        """
        Returns the position of the car on the track axis orthogonal from its
        track coordinates.
        """
        assert np.shape(coordinates) == (3,)

        return coordinates[2]

    def progress(self, coordinates):
        """
        Returns the progress of the car along the track from its track
        coordinates.
        """
        assert np.shape(coordinates) == (3,)

        if coordinates[1] > 0:
            pp = np.arccos(coordinates[0])/(2*math.pi)
        else:
            pp = 1-np.arccos(coordinates[0])/(2*math.pi)
        if pp == 1:
            pp = 0

        return pp

    def coordinates_from_progress(self, progress, position):
        return np.array([
            np.cos(2*math.pi*progress),
            np.sin(2*math.pi*progress),
            position,
        ])

    def invert(self, coordinates):
        assert np.shape(coordinates) == (3,)
        assert coordinates[0] >= -1 and coordinates[0] <= 1
        assert coordinates[1] >= -1 and coordinates[1] <= 1

        if coordinates[1] > 0:
            pp = np.arccos(coordinates[0])/(2*math.pi)
        else:
            pp = 1-np.arccos(coordinates[0])/(2*math.pi)
        if pp == 1:
            pp = 0

        l = pp * len(self.points)
        p = self.points[int(math.floor(l))]

        i = int(math.ceil(l))
        if i == int(math.floor(l)):
            i += 1
        if i == len(self.points):
            i = 0
        n = self.points[i]

        k = l - math.floor(l)

        u = (n-p) / np.linalg.norm(n-p)
        v = np.copy(u)
        v[0] = u[2]
        v[2] = -u[0]

        return ((1-k)*p + k*n + coordinates[2] * self.track_width * v)

    def serialize(self):
        serialized = ''
        for p in self.points:
            serialized += ','.join(map(str, p)) + ';'
        return serialized


class ScriptElemState(Enum):
    STRAIGHT = 1
    CURVE = 2
    ANGLE = 3

class ScriptElem:
    def __init__(self, state, value, count):
        self.state = state
        self.value = value
        self.count = count

class Script:
    def __init__(self, filename):
        self.elems = []
        self.span = 1.0
        self.width = 1.0

        with open(filename) as f:
            for line in f:
                tokens = line.strip().split(" ")

                command = tokens[0]
                arg = tokens[1]

                # print("{}-{}".format(command, arg))

                e = None
                # GPS value define span distance.
                if command == "GSP":
                    self.span = float(arg)
                # GTW value define track (half) width.
                if command == "GTW":
                    self.width = float(arg)
                # DY define the angle difference between two spans in degreee.
                if command == "DY":
                    e = ScriptElem(ScriptElemState.ANGLE, float(arg), 0)
                # L number of spans rotating to the left.
                if command == "L":
                    e = ScriptElem(ScriptElemState.CURVE, -1.0, int(arg))
                # L number of spans rotating to the right.
                if command == "R":
                    e = ScriptElem(ScriptElemState.CURVE, 1.0, int(arg))
                # S is go straingth for N span.
                if command == "S":
                    e = ScriptElem(ScriptElemState.STRAIGHT, 0, int(arg))

                if e is not None:
                    self.elems.append(e)

    def points(self):
        dY = 0.0
        angle = 0.0

        s = np.array((0, 0, 0))
        span = np.array((0, 0, self.span))

        points = []

        for se in self.elems:
            if se.state == ScriptElemState.ANGLE:
                angle = se.value
            if se.state == ScriptElemState.CURVE:
                dY = se.value * angle
                angle = 0.0
            if se.state == ScriptElemState.STRAIGHT:
                dY = 0.0
                angle = 0.0

            for i in range(se.count):
                points.append(np.copy(s))
                span = np.dot(
                    rot_y(np.radians(dY)),
                    span,
                ) / np.linalg.norm(span) * self.span
                s = s + span

        return points

if __name__ == "__main__":
    track = Track("renault_digital_season3")
