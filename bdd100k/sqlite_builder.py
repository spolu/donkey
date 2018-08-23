import argparse
import json
import os
import sqlite3
import sys

def store_data(data_dir, sqlite_path):
    assert os.path.isdir(data_dir)
    conn = sqlite3.connect(sqlite_path)

    c = conn.cursor()
    c.execute('''
DROP TABLE IF EXISTS labels
    ''')
    c.execute('''
CREATE TABLE labels
(name text, dataset text, weather text, scene text, timeofday text, active text)
    ''')
    conn.commit()

    t = 0
    c = conn.cursor()
    for f in os.listdir(os.path.join(data_dir, 'labels/100k/train')):
        with open(os.path.join(data_dir, 'labels/100k/train/' + f), "r") as f:
            label = json.load(f)
            c.execute('''
INSERT INTO labels VALUES ('{}', '{}', '{}', '{}', '{}', '{}')
            '''.format(
                label['name'],
                'train',
                label['attributes']['weather'],
                label['attributes']['scene'],
                label['attributes']['timeofday'],
                'false',
            ))
        t += 1
        if t % 1000 == 0:
            sys.stdout.write('t')
            sys.stdout.flush()
    conn.commit()

    v = 0
    for f in os.listdir(os.path.join(data_dir, 'labels/100k/val')):
        with open(os.path.join(data_dir, 'labels/100k/val/' + f), "r") as f:
            label = json.load(f)
            c.execute('''
INSERT INTO labels VALUES ('{}', '{}', '{}', '{}', '{}', '{}')
            '''.format(
                label['name'],
                'val',
                label['attributes']['weather'],
                label['attributes']['scene'],
                label['attributes']['timeofday'],
                'false',
            ))
        v += 1
        if v % 1000 == 0:
            sys.stdout.write('v')
            sys.stdout.flush()
    conn.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--bdd100k_data_dir', type=str, help="path to bdd100k data dir")
    parser.add_argument('--bdd100k_sqlite_path', type=str, help="path to bdd100k sqlite database")

    args = parser.parse_args()

    assert args.bdd100k_data_dir is not None
    assert args.bdd100k_sqlite_path is not None

    store_data(args.bdd100k_data_dir, args.bdd100k_sqlite_path)
