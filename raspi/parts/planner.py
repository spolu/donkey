import time
from planner import Planner as Planr

class Planner:
    '''
    Installation:
    sudo apt-get install sense-hat

    '''

    def __init__(self, config, poll_delay=0.0166):
        self.on = True
        self.poll_delay = poll_delay
        self.planner = Planr(config)
        self.steering = 1.0
        self.throttle = 1.0

    def update(self):
        while self.on:
            # self.planner.plan()
            time.sleep(self.poll_delay)

    def run_threaded(self,track_progress, track_position, track_angle):
        self.steering, self.throttle = self.planner.plan(track_progress, track_position, track_angle)
        return self.steering, self.throttle

    def run(self,track_progress, track_position, track_angle):
        self.steering, self.throttle = self.planner.plan(track_progress, track_position, track_angle)
        return self.steering, self.throttle

    def shutdown(self):
        self.on = False

if __name__ == "__main__":
    iter = 0
    while iter < 100:
        iter += 1
