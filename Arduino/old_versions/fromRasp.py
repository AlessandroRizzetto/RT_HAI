
import serial
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
    # COM4 is the port number of the Arduino
    ser = serial.Serial('COM4', 9600, timeout=1)
    ser.flush()
    ser.reset_input_buffer()

    while True:
        if ser.in_waiting > 0:  # if there is data in the serial buffer
            line = ser.readline().decode('utf-8').rstrip()
            # to read a part of the line
            # firstWord = ser.readline().decode('utf-8').rstrip().split()[0] # 1st word
            # write to serial
        print("Input 1 to turn ON vibration, 0 to turn OFF vibration")
        input = "1"

        if input == "1":
            ser.write("VIBRATION_ON\n".encode('utf-8'))
            print("Vibration has been turned ON!")
        elif input == "0":
            ser.write("VIBRATION_OFF\n".encode('utf-8'))
            print("Vibration has been turned OFF!")
        break


if __name__ == '__main__':
    main()
