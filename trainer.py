import sys
import random
import argparse
import signal
import os.path

import torch

from coverage import *
from config import *

from models.generative_recurrent import *
from models.competing_recurrent import *
from models.self_play_recurrent import *
from models.a2c_recurrent import *
from models.a2c_coverage import *

def run(main_args):
    cfg = Config(main_args.config_path)

    torch.manual_seed(cfg.get('seed'))
    random.seed(cfg.get('seed'))

    if cfg.get('cuda'):
        torch.cuda.manual_seed(cfg.get('seed'))

    if cfg.get('model') == 'a2c':
        model = A2C(cfg)
    else:
        raise Exception("Unknown model: {}".format(cfg.get('model')))

    episode = 0
    model.initialize()

    while True:
        running = model.batch_train(coverage)

        print('STAT %d %f' % (
            episode,
            running,
        ))
        sys.stdout.flush()

        episode += 1

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('config_path', type=str, help="Path to the config file")

    args = parser.parse_args()

    run(args)
