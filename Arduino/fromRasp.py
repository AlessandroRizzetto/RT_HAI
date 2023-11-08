#!/usr/bin/env python
import serial  # pyserial
import logging

#####
import os
import datetime
import sys
import time
import subprocess
####


def main():

    # --- ARDUINO-PC SERIAL COMMUNICATION SECTION --
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    ser.reset_input_buffer()

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            # to read a part of the line
            # firstWord = ser.readline().decode('utf-8').rstrip().split()[0] # 1st word


if __name__ == '__main__':
    main()
