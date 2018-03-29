from utils import Config
import donkey

import os

os.environ['SIMULATION_PORT'] = '9999'

if __name__ == "__main__":
    cfg = Config('configs/test.json')
    d = donkey.Donkey(cfg)
    d.simulation.launch = False
    d.simulation.start();
    d.reset()