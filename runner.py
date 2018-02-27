import sys
import random
import argparse
import signal
import os.path
import torch

from utils import Config, str2bool
from models.a2c import A2C
from models.random import Random

def run(args):
    cfg = Config(args.config_path)

    cfg.override('worker_count', 1)
    cfg.override('cuda', False)

    if args.headless != None:
        cfg.override('headless', args.headless)

    torch.manual_seed(cfg.get('seed'))
    random.seed(cfg.get('seed'))

    if not args.load_dir:
        raise Exception("Required argument: --load_dir")

    if cfg.get('cuda'):
        torch.cuda.manual_seed(cfg.get('seed'))

    if cfg.get('model') == 'a2c':
        model = A2C(cfg, None, args.load_dir)
    elif cfg.get('model') == 'random':
        model = Random(cfg, None, args.load_dir)
    else:
        raise Exception("Unknown model: {}".format(cfg.get('model')))

    episode = 0
    model.initialize()

    while True:
        reward = model.run()
        print("DONE {}".format(reward))

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('config_path', type=str, help="path to the config file")

    parser.add_argument('--load_dir', type=str, help="path to saved models directory")

    parser.add_argument('--cuda', type=str2bool, help="cuda config override")
    parser.add_argument('--headless', type=str2bool, help="headless config override")
    parser.add_argument('--worker_count', type=int, help="worker_count config override")

    args = parser.parse_args()

    run(args)
