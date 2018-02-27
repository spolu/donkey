import sys
import random
import argparse
import signal
import os.path
import torch

from utils import Config
from models.a2c import A2C
from models.random import Random

def run(args):
    cfg = Config(args.config_path)

    if args.worker_count != None:
        cfg.override('worker_count', args.worker_count)
    if args.cuda != None:
        cfg.override('cuda', args.cuda)
    if args.headless != None:
        cfg.override('headless', args.headless)

    torch.manual_seed(cfg.get('seed'))
    random.seed(cfg.get('seed'))

    if cfg.get('cuda'):
        torch.cuda.manual_seed(cfg.get('seed'))

    if cfg.get('model') == 'a2c':
        model = A2C(cfg, args.save_dir)
    elif cfg.get('model') == 'random':
        model = Random(cfg, args.save_dir)
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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('config_path', type=str, help="path to the config file")
    parser.add_argument('--save_dir', type=str, help="directory to save models")
    parser.add_argument('--cuda', type=str2bool, help="cuda config override")
    parser.add_argument('--headless', type=str2bool, help="headless config override")
    parser.add_argument('--worker_count', type=int, help="worker_count config override")

    args = parser.parse_args()

    run(args)
