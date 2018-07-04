import sys
import random
import argparse
import signal
import os.path
import torch

from utils import Config, str2bool

from reinforce.algorithms import PPO

def run(args):
    cfg = Config(args.config_path)

    if args.worker_count != None:
        cfg.override('worker_count', args.worker_count)
    if args.device != None:
        cfg.override('device', args.device)
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

    if cfg.get('device') != 'cpu':
        torch.cuda.manual_seed(cfg.get('seed'))

    if cfg.get('algorithm') == 'ppo':
        algorithm = PPO(cfg, args.save_dir, args.load_dir)
    assert algorithm is not None

    episode = 0
    algorithm.initialize()

    while True:
        running = algorithm.batch_train()

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

    parser.add_argument('--save_dir', type=str, help="directory to save policies to")
    parser.add_argument('--load_dir', type=str, help="path to saved policies directory")

    parser.add_argument('--device', type=str, help="config override")
    parser.add_argument('--worker_count', type=int, help="config override")

    parser.add_argument('--simulation_headless', type=str2bool, help="config override")
    parser.add_argument('--simulation_time_scale', type=float, help="config override")
    parser.add_argument('--simulation_step_interval', type=float, help="config override")
    parser.add_argument('--simulation_capture_frame_rate', type=int, help="config override")

    args = parser.parse_args()

    run(args)
