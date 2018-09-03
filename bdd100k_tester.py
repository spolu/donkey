import sys
import json
import argparse
import os.path
import os
import cv2
import random
import sqlite3

from utils import Config, str2bool

import numpy as np

import torch
import torch.utils.data as data

from bdd100k import BDD100kBBox

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('config_path', type=str, help="path to the config file")

    parser.add_argument('--bdd100k_data_dir', type=str, help="path to bdd100k data dir")
    parser.add_argument('--bdd100k_sqlite_path', type=str, help="path to bdd100k sqlite database")

    parser.add_argument('--device', type=str, help="config override")

    args = parser.parse_args()

    cfg = Config(args.config_path)

    if args.device != None:
        cfg.override('device', args.device)
    if args.bdd100k_data_dir != None:
        cfg.override('bdd100k_data_dir', args.bdd100k_data_dir)
    if args.bdd100k_sqlite_path != None:
        cfg.override('bdd100k_sqlite_path', args.bdd100k_sqlite_path)

    bbox = BDD100kBBox(cfg, validation=True)
    print("LEN {}".format(bbox.__len__()))
    bbox.__getitem__(1)
