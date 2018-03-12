import sys
import random
import argparse
import signal
import os.path
import torch

from utils import Config, str2bool

def run(args):
    cfg = Config(args.config_path)

    cfg.override('worker_count', 1)
    cfg.override('cuda', False)

    if args.simulation_headless != None:
        cfg.override('simulation_headless', args.simulation_headless)
    if args.simulation_time_scale != None:
        cfg.override('simulation_time_scale', args.simulation_time_scale)
    if args.simulation_step_interval != None:
        cfg.override('simulation_step_interval', args.simulation_step_interval)
    if args.simulation_capture_frame_rate != None:
        cfg.override('simulation_capture_frame_rate', args.simulation_capture_frame_rate)

    torch.manual_seed(cfg.get('seed'))
    random.seed(cfg.get('seed'))

    if not args.load_dir:
        raise Exception("Required argument: --load_dir")

    if cfg.get('cuda'):
        torch.cuda.manual_seed(cfg.get('seed'))

    module = __import__('policies.' + cfg.get('policy'))
    policy = getattr(module, cfg.get('policy')).Policy(cfg)

    module = __import__('models.' + cfg.get('model'))
    model = getattr(module, cfg.get('model')).Model(cfg, policy, None, args.load_dir)

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

    parser.add_argument('--simulation_headless', type=str2bool, help="config override")
    parser.add_argument('--simulation_time_scale', type=float, help="config override")
    parser.add_argument('--simulation_step_interval', type=float, help="config override")
    parser.add_argument('--simulation_capture_frame_rate', type=int, help="config override")

    args = parser.parse_args()

    run(args)
