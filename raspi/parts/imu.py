import time

class Mpu6050:
    '''
    Installation:
    sudo apt install python3-smbus
    or
    sudo apt-get install i2c-tools libi2c-dev python-dev python3-dev
    git clone https://github.com/pimoroni/py-smbus.git
    cd py-smbus/library
    python setup.py build
    sudo python setup.py install

    pip install mpu6050-raspberrypi
    '''

    def __init__(self, addr=0x68, poll_delay=0.0166):
        from mpu6050 import mpu6050
        self.sensor = mpu6050(addr)
        self.accel = { 'x' : 0., 'y' : 0., 'z' : 0. }
        self.gyro = { 'x' : 0., 'y' : 0., 'z' : 0. }
        self.stack = []
        self.temp = 0.
        self.poll_delay = poll_delay
        self.on = True

    def update(self):
        while self.on:
            self.poll()
            time.sleep(self.poll_delay)

    def poll(self):
        self.accel, self.gyro, self.temp = self.sensor.get_all_data()
        self.stack.append[{
            'time': time.time(),
            'temp': self.temp,
            'accel': self.accel,
            'gyro': self.gyro,
        }]

    def run_threaded(self):
        stack = self.stack
        self.stack = []
        return self.accel, self.gyro, stack

    def run(self):
        self.poll()

        stack = self.stack
        self.stack = []
        return self.accel, self.gyro, stack

    def shutdown(self):
        self.on = False


if __name__ == "__main__":
    iter = 0
    p = Mpu6050()
    while iter < 100:
        data = p.run()
        print(data)
        time.sleep(0.1)
        iter += 1
