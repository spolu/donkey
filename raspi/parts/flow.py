import spidev
import time
# import RPi.GPIO as GPIO

# GPIO
PIN_MISO 	= 9
PIN_MOSI  	= 10
PIN_SCK   	= 11
PIN_RESET 	= 22
PIN_CHIPSELECT = 7
ADNS_PRODUCT_ID = 0x00 
ADNS_PRODUCT_ID_VAL = 0x17 
ADNS3080_CONFIGURATION_BITS = 0x0a
ADNS3080_MOTION_BURST = 0x50
ADNS3080_FRAME_CAPTURE  = 0x13
ADNS3080_PIXEL_BURST = 0x40

ADNS3080_PIXELS_X   = 30
ADNS3080_PIXELS_Y   = 30
ADNS3080_PIXEL_SUM  = 0x06
ADNS3080_MOTION  = 0x02
ADNS3080_MOTION_CLEAR = 0x12


class OpticalFlow:
    def __init__(self, addr=0x68, poll_delay=0.05):
        from mpu6050 import mpu6050
        self.sensor = ADNS3080(addr)
        self.speed = { 'dx' : 0., 'dy' : 0.}
        self.poll_delay = poll_delay
        self.on = True

    def update(self):
        while self.on:
            self.poll()
            time.sleep(self.poll_delay)

    def poll(self):
        dx, dy = self.sensor.mousecam_read_motion()
        dx = dx + self.speed['dx']
        dy = dy +  self.speed['dy']
        self.speed = dx, dy

    def run_threaded(self):
        old_speed = self.speed 
        self.speed = { 'dx' : 0., 'dy' : 0.}
        return old_speed

    def run(self):
        dx, dy = self.sensor.mousecam_read_motion() 
        return dx, dy

    def shutdown(self):
        self.on = False


class ADNS3080:

    def __init__(self, addr=0x68):
        # init on arduino
        self.spi = spidev.SpiDev()
        self.spi.open(0, 1)
        self.spi.lsbfirst = False
        self.spi.max_speed_hz = 38400
        self.spi.cshigh = False #has to be false otherwise it doesn't work

        pid = self.mousecam_read_reg(ADNS_PRODUCT_ID)
        assert(pid == [ADNS_PRODUCT_ID_VAL])
        print("initializing CJMCU110...")
        self.mousecam_write_reg(ADNS3080_MOTION_CLEAR, 0xFF) 
        print("clearing motion values...")

        # turn on sensitive mode	
        self.mousecam_write_reg(ADNS3080_CONFIGURATION_BITS, 0x09) 
        # 0x09 is 400 counts resolution
        # 0x19 is 1600 counts resolution

    def mousecam_read_reg(self, reg):
        first = self.spi.xfer([reg], 0, 75, 8)
        ret = self.spi.xfer([255], 0, 1, 8)
        return ret

    def mousecam_write_reg(self, reg, val):
        self.spi.xfer([reg | 0x80])
        # self.spi.xfer([reg])
        self.spi.xfer([val], 0, 50, 8)

        # GPIO.output(PIN_CHIPSELECT, 0)
        return

    def mousecam_read_motion(self):
        val, motion, dx, dy, squal, shutter_lower, shutter_upper, max_pix = self.spi.xfer2([ADNS3080_MOTION_BURST,255, 255, 255, 255, 255, 255, 255])        
        self.md  = { 'motion' : motion, 'dx' : dx, 'dy' : dy, 'squal' : squal, 'shutter_lower' : shutter_lower, 'shutter_upper' : shutter_upper, 'max_pix' : max_pix}
        print("(" + str(dx), "," + str(dy) + ") " + str(squal))
        return dx, dy

    def bytes_to_hex(self,Bytes):
        return ''.join(["0x%02X " % x for x in Bytes]).strip()



if __name__ == "__main__":
    iter = 0
    cimcu = CJMCU110()
    while iter < 100:
        val = cimcu.mousecam_read_motion()
        time.sleep(0.1)
        iter += 1
# 