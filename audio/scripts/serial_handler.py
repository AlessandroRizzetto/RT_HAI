import serial

def connect_serial(port, baudrate):
    try: # to check if the arduino is connected to the PC and manage errors related to the serial communication
        ser = serial.Serial(port, baudrate, timeout=1)
        ser.flush()
        ser.reset_input_buffer()
        return ser
    except serial.SerialException as e:
        print("The Arduino is not connected to the PC")
        return None

def close_serial(ser):
    ser.close()
    
def send_data(ser, data):
    ser.write(f"{data}\n".encode('utf-8'))