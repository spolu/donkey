import time

class Localizer:
    '''
    Installation:
    sudo apt-get install sense-hat

    '''

    def __init__(self, cfg, policy, load_dir ,poll_delay=0.0166):
        self.on = True
        self.poll_delay = poll_delay

    def update(self):
        while self.on:
            time.sleep(self.poll_delay)

    def run_threaded(self, img_array = None,):
        return 1.0,0.1,0.0

    def run(self, img_array = None,):
        return 1.0,0.1,0.0

    def shutdown(self):
        self.on = False

if __name__ == "__main__":
    iter = 0
    while iter < 100:
        iter += 1
