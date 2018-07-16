import sys
import random
import argparse
import signal
import os.path
import torch

from utils import Config, str2bool

from synthetic import Synthetic

def run(args):
    cfg = Config(args.config_path)

    if args.device != None:
        cfg.override('device', args.device)

    torch.manual_seed(cfg.get('seed'))
    random.seed(cfg.get('seed'))
    if cfg.get('device') != 'cpu':
        torch.cuda.manual_seed(cfg.get('seed'))

    if not args.train_capture_set_dir:
        raise Exception("Required argument: --train_capture_set_dir")
    if not args.test_capture_set_dir:
        raise Exception("Required argument: --test_capture_set_dir")

    synthetic = Synthetic(cfg, args.save_dir, args.load_dir)

    episode = 0

    synthetic.generator_initialize_training(
        args.train_capture_set_dir, args.test_capture_set_dir,
    )

    while True:
        loss = synthetic.generator_batch_train()

        print("BATCH_TRAIN {} {:.5f} {:.5f} {:.5f}".format(
            episode,
            loss.avg,
            loss.min,
            loss.max,
        ))
        sys.stdout.flush()

        loss = synthetic.generator_batch_test()
        print("BATCH_TEST {} {:.5f} {:.5f} {:.5f}".format(
            episode,
            loss.avg,
            loss.min,
            loss.max,
        ))
        sys.stdout.flush()

        episode += 1
if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('config_path', type=str, help="path to the config file")

    parser.add_argument('--save_dir', type=str, help="directory to save policies to")
    parser.add_argument('--load_dir', type=str, help="path to saved policies directory")

    parser.add_argument('--device', type=str, help="config override")

    parser.add_argument('--train_capture_set_dir', type=str, help="path to train capture set")
    parser.add_argument('--test_capture_set_dir', type=str, help="path to test capture set")

    args = parser.parse_args()

    run(args)
