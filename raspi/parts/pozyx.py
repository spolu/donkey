import time

from pypozyx import (POZYX_POS_ALG_UWB_ONLY, POZYX_3D, Coordinates,
                     POZYX_SUCCESS, POZYX_ANCHOR_SEL_AUTO, DeviceCoordinates,
                     PozyxSerial, get_first_pozyx_serial_port, SingleRegister,
                     DeviceList)
from pythonosc.udp_client import SimpleUDPClient

class Pozyxer:
    def __init__(self):
        # shortcut to not have to find out the port yourself
        serial_port = get_first_pozyx_serial_port()
        if serial_port is None:
            print("No Pozyx connected. Check your USB cable or your driver!")
            quit()

        # self.remote_id = 0x6069           # remote device network ID
        # self.remote_id = 0x677d           # remote device network ID
        self.remote_id = None               # not remote

        use_processing = True        # enable to send position data through OSC
        ip = "127.0.0.1"             # IP for the OSC UDP
        network_port = 8888          # network port for the OSC UDP
        self.osc_udp_client = None

        if use_processing:
            osc_udp_client = SimpleUDPClient(ip, network_port)

        # Necessary data for calibration, change the IDs and coordinates yourself
        self.anchors = [
            DeviceCoordinates(0x6e67, 1, Coordinates(700, -570, 2600)),
            DeviceCoordinates(0x6e64, 1, Coordinates(-3110, -1325, 1320)),
            DeviceCoordinates(0x6940, 1, Coordinates(-3110, -985, -1760)),
            DeviceCoordinates(0x6935, 1, Coordinates(700, -570, -10225)),
        ]

        self.algorithm = POZYX_POS_ALG_UWB_ONLY     # positioning algorithm to use
        # self.dimension = POZYX_2_5D               # positioning dimension
        self.dimension = POZYX_3D
        self.height = 17                            # height of device, required in 2.5D positioning

        self.pozyx = PozyxSerial(serial_port)

        self.pozyx.clearDevices(self.remote_id)
        self.setAnchorsManual()

        self.position = None
        self.stack = []
        self.on = True

    def setAnchorsManual(self):
        """Adds the manually measured anchors to the Pozyx's device list one for one."""
        status = self.pozyx.clearDevices(self.remote_id)
        for anchor in self.anchors:
            status &= self.pozyx.addDevice(anchor, self.remote_id)
        if len(self.anchors) > 4:
            status &= self.pozyx.setSelectionOfAnchors(POZYX_ANCHOR_SEL_AUTO, len(self.anchors))
        return status

    def poll(self):
        self.pozyx_position = Coordinates()
        status = self.pozyx.doPositioning(
            self.pozyx_position, self.dimension, self.height, self.algorithm,
            remote_id=self.remote_id,
        )
        if status == POZYX_SUCCESS:
            self.position = self.pozyx_position
            self.stack.append({
                'time': time.time(),
                'positon': {
                    'x': float(self.position.x)/1000.0,
                    'y': float(self.position.y)/1000.0,
                    'z': float(self.position.z)/1000.0,
                },
            })

    def update(self):
        while self.on:
            self.poll()

    def run(self):
        self.poll()

        stack = self.stack
        self.stack = []

        return {
            'x': float(self.position.x)/1000.0,
            'y': float(self.position.y)/1000.0,
            'z': float(self.position.z)/1000.0,
        }, stack

    def run_threaded(self):
        stack = self.stack
        self.stack = []

        return {
            'x': float(self.position.x)/1000.0,
            'y': float(self.position.y)/1000.0,
            'z': float(self.position.z)/1000.0,
        }, stack

    def shutdown(self):
        self.on = False


# class ReadyToLocalize(object):
#     """Continuously calls the Pozyx positioning function and prints its position."""
#
#     def __init__(self, pozyx, osc_udp_client, anchors, algorithm=POZYX_POS_ALG_UWB_ONLY, dimension=POZYX_3D, height=1000, remote_id=None):
#         self.pozyx = pozyx
#         self.osc_udp_client = osc_udp_client
#
#         self.anchors = anchors
#         self.algorithm = algorithm
#         self.dimension = dimension
#         self.height = height
#         self.remote_id = remote_id
#
#     def setup(self):
#         """Sets up the Pozyx for positioning by calibrating its anchor list."""
#         self.pozyx.clearDevices(self.remote_id)
#         self.setAnchorsManual()
#         # self.printPublishConfigurationResult()
#
#     def loop(self):
#         """Performs positioning and displays/exports the results."""
#         position = Coordinates()
#         status = self.pozyx.doPositioning(
#             self.position, self.dimension, self.height, self.algorithm, remote_id=self.remote_id)
#         if status == POZYX_SUCCESS:
#             self.printPublishPosition(position)
#         else:
#             self.printPublishErrorCode("positioning")
#
#     def printPublishPosition(self, position):
#         """Prints the Pozyx's position and possibly sends it as a OSC packet"""
#         network_id = self.remote_id
#         if network_id is None:
#             network_id = 0
#         print("POS ID {}, x(mm): {pos.x} y(mm): {pos.y} z(mm): {pos.z}".format(
#             "0x%0.4x" % network_id, pos=position))
#         if self.osc_udp_client is not None:
#             self.osc_udp_client.send_message(
#                 "/position", [network_id, int(position.x), int(position.y), int(position.z)])
#
#     def printPublishErrorCode(self, operation):
#         """Prints the Pozyx's error and possibly sends it as a OSC packet"""
#         error_code = SingleRegister()
#         network_id = self.remote_id
#         if network_id is None:
#             self.pozyx.getErrorCode(error_code)
#             print("LOCAL ERROR %s, %s" % (operation, self.pozyx.getErrorMessage(error_code)))
#             if self.osc_udp_client is not None:
#                 self.osc_udp_client.send_message("/error", [operation, 0, error_code[0]])
#             return
#         status = self.pozyx.getErrorCode(error_code, self.remote_id)
#         if status == POZYX_SUCCESS:
#             print("ERROR %s on ID %s, %s" %
#                   (operation, "0x%0.4x" % network_id, self.pozyx.getErrorMessage(error_code)))
#             if self.osc_udp_client is not None:
#                 self.osc_udp_client.send_message(
#                     "/error", [operation, network_id, error_code[0]])
#         else:
#             self.pozyx.getErrorCode(error_code)
#             print("ERROR %s, couldn't retrieve remote error code, LOCAL ERROR %s" %
#                   (operation, self.pozyx.getErrorMessage(error_code)))
#             if self.osc_udp_client is not None:
#                 self.osc_udp_client.send_message("/error", [operation, 0, -1])
#             # should only happen when not being able to communicate with a remote Pozyx.
#
#     def printPublishConfigurationResult(self):
#         """Prints and potentially publishes the anchor configuration result in a human-readable way."""
#         list_size = SingleRegister()
#
#         self.pozyx.getDeviceListSize(list_size, self.remote_id)
#         print("List size: {0}".format(list_size[0]))
#         if list_size[0] != len(self.anchors):
#             self.printPublishErrorCode("configuration")
#             return
#         device_list = DeviceList(list_size=list_size[0])
#         self.pozyx.getDeviceIds(device_list, self.remote_id)
#         print("Calibration result:")
#         print("Anchors found: {0}".format(list_size[0]))
#         print("Anchor IDs: ", device_list)
#
#         for i in range(list_size[0]):
#             anchor_coordinates = Coordinates()
#             self.pozyx.getDeviceCoordinates(device_list[i], anchor_coordinates, self.remote_id)
#             print("ANCHOR, 0x%0.4x, %s" % (device_list[i], str(anchor_coordinates)))
#             if self.osc_udp_client is not None:
#                 self.osc_udp_client.send_message(
#                     "/anchor", [device_list[i], int(anchor_coordinates.x), int(anchor_coordinates.y), int(anchor_coordinates.z)])
#                 time.sleep(0.025)
#
#     def printPublishAnchorConfiguration(self):
#         """Prints and potentially publishes the anchor configuration"""
#         for anchor in self.anchors:
#             print("ANCHOR,0x%0.4x,%s" % (anchor.network_id, str(anchor.coordinates)))
#             if self.osc_udp_client is not None:
#                 self.osc_udp_client.send_message(
#                     "/anchor", [anchor.network_id, int(anchor.coordinates.x), int(anchor.coordinates.y), int(anchor.coordinates.z)])
#                 time.sleep(0.025)

if __name__ == "__main__":
    iter = 0
    p = Pozyxer()
    while iter < 100:
        position = p.run()
        print(position)
        time.sleep(0.1)
        iter += 1
