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

class CJMCU110:

    def __init__(self, addr=0x68):
        # init on arduino
        self.spi = spidev.SpiDev()
        self.spi.open(0, 1)
        self.spi.lsbfirst = False
        self.spi.max_speed_hz = 38400
        self.spi.cshigh = False #has to be true otherwise it doesn't work
        # self.spi.max_speed_hz = 400
        # self.spi.loop = False
        # 

        # self.mousecam_reset()
        pid = self.mousecam_read_reg(ADNS_PRODUCT_ID)
        assert(pid == [ADNS_PRODUCT_ID_VAL])
        print("initializing CJMCU110...")
        # self.mousecam_write_reg(ADNS3080_MOTION_CLEAR, 0xFF) 
        # print("clearing motion values...")


        # turn on sensitive mode	
        self.mousecam_write_reg(ADNS3080_CONFIGURATION_BITS, 0x09) 
        # 0x09 is 400 counts resolution
        # 0x19 is 1600 counts resolution


    def mousecam_read_reg(self, reg):
        first = self.spi.xfer([reg], 0, 75, 8)
        print("{0:b}".format(first[0]))
        ret = self.spi.xfer([255], 0, 1, 8)
        print("{0:b}".format(ret[0]))
        return ret

    def mousecam_write_reg(self, reg, val):
        self.spi.xfer([reg | 0x80])
        # self.spi.xfer([reg])
        self.spi.xfer([val], 0, 50, 8)

        # GPIO.output(PIN_CHIPSELECT, 0)
        return

    def mousecam_read_motion(self):
        self.spi.xfer([ADNS3080_MOTION_BURST], 0, 75, 8)
        motion, dx, dy, squal, shutter_lower, shutter_upper, max_pix = self.spi.xfer([255, 255, 255, 255, 255, 255, 255])        
        
        self.md  = { 'motion' : motion, 'dx' : dx, 'dy' : dy, 'squal' : squal, 'shutter_lower' : shutter_lower, 'shutter_upper' : shutter_upper, 'max_pix' : max_pix}
        print(self.md)

    # def mousecam_frame_capture(self):
    #     self.mousecam_write_reg(ADNS3080_FRAME_CAPTURE, 0x83)

    #     self.spi.xfer([ADNS3080_PIXEL_BURST], 38400, 50, 8)
    #     pix = 0
    #     started = 0x00
    #     count = 0
    #     timeout = 0
    #     ret = 0
    #     pdata = []
    #     while count < ADNS3080_PIXELS_X * ADNS3080_PIXELS_Y:
    #         pix = self.spi.xfer([0xff], 38400, 10, 8)
    #         if started == 0:
    #             print(pix)
    #             if (pix[0] & 0x40):
    #                 print("has started")
    #                 started = 1
    #             else:
    #                 timeout += 1
    #                 if (timeout == 100):
    #                     ret = -1
    #                     break
    #         elif started == 1:
    #             pdata[count] = pix
    #             count += 1

    #     return pdata


    def shutdown(self):
        pass
        # GPIO.cleanup()



if __name__ == "__main__":
    iter = 0
    cimcu = CJMCU110()
    while iter < 100:
        # pdata = cimcu.mousecam_frame_capture()     
        # print(pdata)   

        #tell if motion has occured
        val = cimcu.mousecam_read_motion()
        # val = cimcu.mousecam_read_reg(ADNS3080_MOTION) 
        # print("{0:b}".format(val[0]))
        # cimcu.mousecam_read_motion()
        time.sleep(1)
        iter += 1

    # finally:
        # GPIO.cleanup()

# spi = spidev.SpiDev()
# spi.open(0, 0)
# spi.max_speed_hz = 7629

# # Split an integer input into a two byte array to send via SPI
# def write_pot(input):
#     msb = input &gt;&gt; 8
#     lsb = input &amp; 0xFF
#     spi.xfer([msb, lsb])

# # Repeatedly switch a MCP4151 digital pot off then on
# while True:
#     write_pot(0x1FF)
#     time.sleep(0.5)
#     write_pot(0x00)
#     time.sleep(0.5)
# 