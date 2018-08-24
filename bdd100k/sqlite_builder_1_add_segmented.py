import argparse
import json
import os
import sqlite3
import sys

def update_data(data_dir, sqlite_path):
    assert os.path.isdir(data_dir)
    conn = sqlite3.connect(sqlite_path)

#     c = conn.cursor()
#     c.execute('''
# ALTER TABLE labels ADD COLUMN segmented text
#     ''')
#     conn.commit()

    c = conn.cursor()
    c.execute('''
UPDATE labels SET segmented='false'
    ''')
    conn.commit()

    t = 0
    c = conn.cursor()
    for f in os.listdir(os.path.join(data_dir, 'seg/labels/train')):
        c.execute('''
 UPDATE labels SET segmented='true' WHERE name=?
        ''', [f[:17]])
        t += 1
    conn.commit()
    print("updated {} train".format(t))

    v = 0
    c = conn.cursor()
    for f in os.listdir(os.path.join(data_dir, 'seg/labels/val')):
        c.execute('''
UPDATE labels SET segmented='true' WHERE name=?
        ''', [f[:17]])
        v += 1
    conn.commit()
    print("updated {} train".format(v))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--bdd100k_data_dir', type=str, help="path to bdd100k data dir")
    parser.add_argument('--bdd100k_sqlite_path', type=str, help="path to bdd100k sqlite database")

    args = parser.parse_args()

    assert args.bdd100k_data_dir is not None
    assert args.bdd100k_sqlite_path is not None

    update_data(args.bdd100k_data_dir, args.bdd100k_sqlite_path)
