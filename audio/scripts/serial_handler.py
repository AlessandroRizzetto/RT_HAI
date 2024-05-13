import serial
from serial.tools import list_ports

def connect_serial(port, baudrate):
    '''
    Create serial connection.

    Args:
        port (string): port name
        baudrate (int): Baud rate such as 9600
    
    Returns:
        ser (serial.Serial,None): serial connection
    '''

    try: # to check if the arduino is connected to the PC and manage errors related to the serial communication
        ser = serial.Serial(port, baudrate, timeout=1)
        ser.flush()
        ser.reset_input_buffer()
        return ser
    except serial.SerialException as e:
        print("The Arduino is not connected to the PC")
        return None

def close_serial(ser):
    '''
    Close serial connection.

    Args:
        ser (serial.Serial): serial connection
    '''

    ser.close()
    
def send_data(ser, data):
    '''
    Send data on serial port.

    Args:
        ser (serial.Serial): serial connection
        data (string): data to send
    '''

    ser.write(f"{data}\n".encode('utf-8'))

# def can_connect(port):
#     '''
#     Check if the specified port is available.
    
#     Args:
#         port (string): port name
    
#     Returns:
#         (tuple,None): port name and description
#     '''
    
#     ports_list = [tuple(p) for p in list(list_ports.comports())]
#     return next((p for p in ports_list if p[0] == port), None)