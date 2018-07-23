import collections

Observation = collections.namedtuple(
    'Observation',
    (
        'time '
        'track_coordinates '
        'track_angles '
        'track_linear_speed '
        'position '
        'velocity '
        'angular_velocity '
        'camera '
        # 'camera_stack '
    ),
)

Command = collections.namedtuple(
    'Command',
    (
        'steering '
        'throttle '
        'brake'
    ),
)

Telemetry = collections.namedtuple(
    'Telemetry',
    (
        'time '
        'camera '
        'position '
        'velocity '
        'angular_velocity'
    ),
)

