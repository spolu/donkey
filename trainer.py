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

    if args.worker_count != None:
        cfg.override('worker_count', args.worker_count)
    if args.cuda != None:
        cfg.override('cuda', args.cuda)
    if args.simulation_headless != None:
        cfg.override('simulation_headless', args.simulation_headless)
    if args.simulation_time_scale != None:
        cfg.override('simulation_time_scale', args.simulation_time_scale)

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

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('config_path', type=str, help="path to the config file")

    parser.add_argument('--save_dir', type=str, help="directory to save models")

    parser.add_argument('--cuda', type=str2bool, help="config override")
    parser.add_argument('--worker_count', type=int, help="config override")
    parser.add_argument('--simulation_headless', type=str2bool, help="config override")
    parser.add_argument('--simulation_time_scale', type=float, help="config override")

    args = parser.parse_args()

    run(args)
