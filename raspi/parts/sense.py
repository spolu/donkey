import time
from sense_hat import SenseHat

class Sense:
    '''
    Installation:
    sudo apt-get install sense-hat

    '''

    def __init__(self, poll_delay=0.0166):
        self.sense = SenseHat()
        self.sense.set_imu_config(True, True, True)
        self.sense.show_message("Dr1vin'")
        self.orientation = { 'p:' : 0., 'r' : 0., 'y' : 0. }
        self.on = True
        self.poll_delay = poll_delay

    def update(self):
        while self.on:
            self.poll()
            time.sleep(self.poll_delay)

    def poll(self):
        self.orientation= self.sense.get_orientation_radians()

    def run_threaded(self):
        # return self.accel['x'], self.accel['y'], self.accel['z'], self.gyro['x'], self.gyro['y'], self.gyro['z'], self.temp
        return self.orientation

    def run(self):
        self.poll()
        # return self.accel['x'], self.accel['y'], self.accel['z'], self.gyro['x'], self.gyro['y'], self.gyro['z'], self.temp
        return self.orientation

    def shutdown(self):
        self.on = False

if __name__ == "__main__":
    iter = 0
    p = Sense()
    while iter < 100:
        data = p.run()
        print(data)
        time.sleep(0.1)
        iter += 1
