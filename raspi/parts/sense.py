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
        self.sense.set_rotation(0)
        self.orientation = { 'p:' : 0., 'r' : 0., 'y' : 0. }
        self.on = True
        self.poll_delay = poll_delay
        
        self.angle = 0.0
        self.throttle = 0.0

    def update(self):
        while self.on:
            self.poll()
            self.show_direction()
            time.sleep(self.poll_delay)

    def poll(self):
        self.orientation= self.sense.get_orientation_radians()

    def run_threaded(self, angle = None, throttle = None):
        self.throttle = throttle
        self.angle = angle
        # return self.accel['x'], self.accel['y'], self.accel['z'], self.gyro['x'], self.gyro['y'], self.gyro['z'], self.temp
        return self.orientation

    def run(self, angle = None, throttle = None):
        self.poll()
        self.throttle = throttle
        self.angle = angle
        # return self.accel['x'], self.accel['y'], self.accel['z'], self.gyro['x'], self.gyro['y'], self.gyro['z'], self.temp
        return self.orientation

    def show_direction(self):
        green = int(127*self.throttle)
        X = [128 - green, 128 + green, 0]
        O = [0, 0, 0]

        pixels = [
            O, O, O, O, X, O, O, O,
            O, O, O, O, O, X, O, O,
            O, O, O, O, O, O, X, O,
            X, X, X, X, X, X, X, X,
            X, X, X, X, X, X, X, X,
            O, O, O, O, O, O, X, O,
            O, O, O, O, O, X, O, O,
            O, O, O, O, X, O, O, O
            ]
        if(self.angle > 0.2):
            pixels = [
            O, O, O, O, O, O, O, O,
            O, X, O, O, O, O, O, O,
            O, O, X, O, O, O, O, O,
            O, O, O, X, O, O, O, O,
            O, O, O, O, X, O, O, X,
            O, O, O, O, O, X, O, X,
            O, O, O, O, O, O, X, X,
            O, O, O, O, X, X, X, X,
            ]
        elif (self.angle < -0.2):
            pixels = [
            O, O, O, O, X, X, X, X,
            O, O, O, O, O, O, X, X,
            O, O, O, O, O, X, O, X.
            O, O, O, O, X, O, O, X,
            O, O, O, X, O, O, O, O,
            O, O, X, O, O, O, O, O,
            O, X, O, O, O, O, O, O,
            O, O, O, O, O, O, O, O
            ]

        self.sense.set_pixels(pixels)

    def shutdown(self):
        self.sense.clear(0,0,0)
        self.on = False

if __name__ == "__main__":
    iter = 0
    p = Sense()
    while iter < 100:
        data = p.run()
        print(data)
        time.sleep(0.1)
        iter += 1
