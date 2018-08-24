import argparse
import eventlet
import eventlet.wsgi
import json
import os
import sqlite3
import sys

from flask import Flask
from flask import jsonify
from flask import abort
from flask import send_file
from flask import request

_app = Flask(__name__)
_conn = None
_data_dir = None

_labels = {}
_attributes = {
    'dataset': [],
    'weather': [],
    'scene': [],
    'timeofday': [],
    'segmented': [],
    'active': [],
}

@_app.route('/attributes', methods=['GET'])
def retrieve_attributes():
    global _attributes

    return jsonify(_attributes)

@_app.route('/videos', methods=['GET'])
def retrieve_videos():
    global _attributes
    global _conn

    query = {}
    for attr in _attributes:
        value = request.args.getlist(attr+'[]')
        if len(value) > 0:
            query[attr] = value
        else:
            query[attr] = _attributes[attr]

    c = _conn.cursor()
    q = '''
SELECT * FROM labels
WHERE
  dataset IN ({})
  AND weather IN ({})
  AND scene IN ({})
  AND timeofday IN ({})
  AND segmented IN ({})
  AND active IN ({})
'''.format(
    ','.join(['?']*len(query['dataset'])),
    ','.join(['?']*len(query['weather'])),
    ','.join(['?']*len(query['scene'])),
    ','.join(['?']*len(query['timeofday'])),
    ','.join(['?']*len(query['segmented'])),
    ','.join(['?']*len(query['active'])),
)
    results = []
    for r in c.execute(
            q,
            query['dataset'] +
            query['weather'] +
            query['scene'] +
            query['timeofday'] +
            query['segmented'] +
            query['active']
    ):
        results += [r]

    return jsonify(results)

@_app.route('/videos/<name>', methods=['POST'])
def update_video(name):
    global _conn

    active = request.get_json()['active']
    c = _conn.cursor()
    c.execute('''
UPDATE labels SET active=? WHERE name=?
    ''', (
        active,
        name,
    ))
    _conn.commit()

    return jsonify([])

@_app.route('/images/<name>.jpg', methods=['GET'])
def retrieve_image(name):
    global _data_dir
    global _conn

    c = _conn.cursor()
    dataset = c.execute('''
SELECT dataset FROM labels WHERE name=?
    ''',(
        name,
    )).fetchone()[0]
    print(dataset)

    return send_file(
        os.path.join(_data_dir, 'images/100k', dataset, name + '.jpg'),
        attachment_filename='%s.jpg'.format(name),
        mimetype='image/jpeg',
    )

@_app.route('/videos/<name>/objects', methods=['GET'])
def retrieve_videos_objects(name):
    global _data_dir
    global _conn

    c = _conn.cursor()
    dataset = c.execute('''
SELECT dataset FROM labels WHERE name=?
    ''',(
        name,
    )).fetchone()[0]
    print(dataset)

    objects = []
    with open(os.path.join(_data_dir, 'labels/100k/' + dataset + '/' + name + '.json'), "r") as f:
        label = json.load(f)
        for o in label['frames'][0]['objects']:
            if o['category'] in [
                    'traffic sign',
                    'car', 'truck', 'bus',
                    'lane/road curb', 'lane/single white', 'lane/single yellow',
            ] and ('direction' not in o['attributes'] or o['attributes']['direction'] == 'parallel'):
                objects.append(o)

    return jsonify(objects)

def setup_attributes():
    global _attributes

    c = _conn.cursor()
    for attr in _attributes:
        for r in c.execute('''
SELECT {}, COUNT(*) FROM labels GROUP BY 1
        '''.format(
            attr,
        )):
            _attributes[attr] += [r[0]]

def run_server():
    global _app

    print("Starting shared server: port=9094")
    address = ('0.0.0.0', 9094)
    try:
        eventlet.wsgi.server(eventlet.listen(address), _app)
    except KeyboardInterrupt:
        print("Stopping shared server")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--bdd100k_data_dir', type=str, help="path to bdd100k data dir")
    parser.add_argument('--bdd100k_sqlite_path', type=str, help="path to bdd100k sqlite database")

    args = parser.parse_args()

    assert args.bdd100k_data_dir is not None
    assert args.bdd100k_sqlite_path is not None

    _conn = sqlite3.connect(args.bdd100k_sqlite_path)
    _data_dir = args.bdd100k_data_dir

    setup_attributes()
    run_server()

# @_app.route('/images/<id>', methods=['GET'])
# def image():
#     return send_file(
#         io.BytesIO(encoded.tobytes()),
#         attachment_filename='%d.jpeg' % index,
#         mimetype='image/jpeg',
#     )
