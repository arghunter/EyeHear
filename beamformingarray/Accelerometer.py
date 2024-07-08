"""
A simple helper class for using the UART-RVC mode of the BNO08x IMUs.

Provides intuitive access to yaw, pitch, roll, x_accel, y_accel and z_accel
Ported to pySerial by Armaan Gomes
Ported to MicroPython by rdagger from Adafruit_CircuitPython_BNO08x_RVC
https://github.com/adafruit/Adafruit_CircuitPython_BNO08x_RVC.git
The original code was written by
Bryan Siepert for Adafruit Industries - Copyright (c) 2020 - MIT License

Implementation Notes
--------------------
    Adafruit 9-DOF Orientation IMU Fusion Breakout - BNO085 (BNO080)
        <https://www.adafruit.com/product/4754>`_ (Product ID: 4754)

    Make sure pin P0 is pulled high to 3.3 V to enable UART-RVC mode
"""
import serial
from struct import unpack_from
from time import time

class RVCReadTimeoutError(Exception):
    """Raised if a UART-RVC message cannot be read before the given timeout."""

class BNO08x_RVC:
    """Reads BNO08x IMUs using the UART-RVC mode."""

    def __init__(self, serial_port, baudrate=115200, timeout=1):
        """Constructor for BNO08x_RVC.

        Args:
            serial_port (str): Serial port name (e.g., 'COM1' for Windows or '/dev/ttyUSB0' for Linux)
            baudrate (int): Baud rate for serial communication
            timeout (int): Serial timeout in seconds
        """
        self._serial = serial.Serial(serial_port, baudrate, timeout=timeout)
        self._timeout = timeout
        self.offsets=(0,0,0,0,0,0)

    @staticmethod
    def _parse_frame(frame,offsets):
        heading_frame = unpack_from("<BhhhhhhBBBB", frame)
        (
            _index,
            yaw,
            pitch,
            roll,
            x_accel,
            y_accel,
            z_accel,
            _res1,
            _res2,
            _res3,
            _checksum,
        ) = heading_frame
        checksum_calc = sum(frame[0:16]) % 256
        if checksum_calc != _checksum:
            return None
        yaw *= 0.01
        pitch *= 0.01
        roll *= 0.01
        x_accel *= 0.0098067
        y_accel *= 0.0098067
        z_accel *= 0.0098067
        # print(yaw)
        return (yaw-offsets[0], pitch-offsets[1], roll-offsets[2], x_accel-offsets[3], y_accel-offsets[4], z_accel-offsets[5])

    @property
    def heading(self):
        """Return the current heading.

        Returns:
            List(float):
               Yaw
               Pitch
               Roll
               X-Axis Acceleration
               Y-Axis Acceleration
               Z-Axis Acceleration
        """
        # try to read initial packet start byte
        data = None
        start_time = time()
        while time() - start_time < self._timeout:
            data = self._serial.read(2)
            # print(data)
            if data is None:
                continue
            if data[0] == 0xAA and data[1] == 0xAA:
                msg = self._serial.read(17)
                heading = self._parse_frame(msg,self.offsets)
                if heading is None:
                    continue
                return heading
        raise RVCReadTimeoutError("Unable to read RVC heading message")
    def get_yaw(self):
        return self.heading[0]
    def zero(self):
        self.offsets=self.heading
        
# Example usage
# bno = BNO08x_RVC(serial_port='/dev/ttyUSB0', baudrate=9600)
# # print(bno.heading)
# accel=BNO08x_RVC('COM4')
# # accel.zero()
# while True:
#     print(accel.get_yaw())


    
    
