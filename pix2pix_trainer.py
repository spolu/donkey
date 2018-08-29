import sys
import random
import argparse
import signal
import os.path
import torch

from utils import Config, str2bool

from pix2pix import Pix2Pix

def run(args):
    cfg = Config(args.config_path)

    if args.device != None:
        cfg.override('device', args.device)
    if args.pix2pix_save_dir != None:
        cfg.override('pix2pix_save_dir', args.pix2pix_save_dir)
    if args.pix2pix_load_dir != None:
        cfg.override('pix2pix_load_dir', args.pix2pix_load_dir)
    if args.bdd100k_data_dir != None:
        cfg.override('bdd100k_data_dir', args.bdd100k_data_dir)
    if args.bdd100k_sqlite_path != None:
        cfg.override('bdd100k_sqlite_path', args.bdd100k_sqlite_path)

    torch.manual_seed(cfg.get('seed'))
    random.seed(cfg.get('seed'))
    if cfg.get('device') != 'cpu':
        torch.cuda.manual_seed(cfg.get('seed'))

    pix2pix = Pix2Pix(cfg)

    episode = 0
    pix2pix.initialize_training()

    while True:
        pix2pix.batch_train()
        episode += 1

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('config_path', type=str, help="path to the config file")

    parser.add_argument('--pix2pix_save_dir', type=str, help="config override")
    parser.add_argument('--pix2pix_load_dir', type=str, help="config override")

    parser.add_argument('--bdd100k_data_dir', type=str, help="config override")
    parser.add_argument('--bdd100k_sqlite_path', type=str, help="config override")

    parser.add_argument('--device', type=str, help="config override")

    args = parser.parse_args()

    run(args)
