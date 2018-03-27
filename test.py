from utils import Config

import donkey

if __name__ == "__main__":
    cfg = Config('configs/test.json')
    d = donkey.Donkey(cfg)
    d.simulation.launch = False
    d.simulation.start();
    d.reset()