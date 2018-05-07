import random
import os

import numpy as np
from enum import Enum

"""
Track interface
"""

# import pdb; pdb.set_trace()

SPAN_DIST = 5.0

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

        # interpolate to first point
        self.points.append([
            0.001 * self.points[-1][0] + 0.999 * self.points[0][0],
            0.001 * self.points[-1][1] + 0.999 * self.points[0][1],
            0.001 * self.points[-1][2] + 0.999 * self.points[0][2],
        ])

        # recompute length
        self.length = 0.0
        for i in range(len(self.points)):
            self.length += np.linalg.norm(
                self.points[self.next(i)] - self.points[i],
            )

    def randomize(self):
        """
        Randomize track by randomly applying symetry to the track and picking
        translating such that the starting location is picked randomly.
        """
        self.points = self.script.points()

        if random.randint(0, 1) == 1:
            for i in range(len(self.points)):
                self.points[i][0] = -self.points[i][0]
        if random.randint(0, 1) == 1:
            for i in range(len(self.points)):
                self.points[i][2] = -self.points[i][2]

        position = random.randint(0, len(self.points)-1)
        for i in range(position):
            v = self.points[1] - self.points[0]
            for j in range(len(self.points)):
                u = [0, 0, 1]
                w = v / np.linalg.norm(v)
                theta = -np.sign(np.linalg.det([
                    u,
                    w,
                    [0,1,0],
                ])) * np.arccos(np.dot(u, w))
                self.points[j] = np.dot(rot_y(theta), self.points[j] - v)
            self.points = self.points[1:] + self.points[:1]

        # interpolate to first point
        self.points.append([
            0.001 * self.points[-1][0] + 0.999 * self.points[0][0],
            0.001 * self.points[-1][1] + 0.999 * self.points[0][1],
            0.001 * self.points[-1][2] + 0.999 * self.points[0][2],
        ])

        # recompute length
        self.length = 0.0
        for i in range(len(self.points)):
            self.length += np.linalg.norm(
                self.points[self.next(i)] - self.points[i],
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

    def position(self, position, offset=0):
        """
        Returns the position of the car on the track axis orthogonal.
        """
        closests = self.closests(position, offset)
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

    def progress(self, position):
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
        return p

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

        with open(filename) as f:
            for line in f:
                tokens = line.strip().split(" ")

                command = tokens[0]
                arg = tokens[1]

                # print("{}-{}".format(command, arg))

                e = None
                if command == "DY":
                    e = ScriptElem(ScriptElemState.ANGLE, float(arg), 0)
                elif command == "L":
                    e = ScriptElem(ScriptElemState.CURVE, -1.0, int(arg))
                elif command == "R":
                    e = ScriptElem(ScriptElemState.CURVE, 1.0, int(arg))
                elif command == "S":
                    e = ScriptElem(ScriptElemState.STRAIGHT, 0, int(arg))

                self.elems.append(e)

    def points(self):
        dY = 0.0
        angle = 0.0

        s = np.array((0, 0, 0))
        span = np.array((0, 0, SPAN_DIST))

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
                ) / np.linalg.norm(span) * SPAN_DIST
                s = s + span

        return points

