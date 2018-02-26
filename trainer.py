import sys
import random
import argparse
import signal
import os.path
import torch

from utils import Config
from models.a2c import A2C

def run(args):
    cfg = Config(args.config_path)

    torch.manual_seed(cfg.get('seed'))
    random.seed(cfg.get('seed'))

    if cfg.get('cuda'):
        torch.cuda.manual_seed(cfg.get('seed'))

    if cfg.get('model') == 'a2c':
        model = A2C(cfg, args.save_dir)
    else:
        raise Exception("Unknown model: {}".format(cfg.get('model')))

    episode = 0
    model.initialize()

    while True:
        running = model.batch_train()

        print('STAT %d %f' % (
            episode,
            running,
        ))
        sys.stdout.flush()

        episode += 1

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('config_path', type=str, help="path to the config file")
    parser.add_argument('--save_dir', type=str, help="directory to save models")

    args = parser.parse_args()

    run(args)
