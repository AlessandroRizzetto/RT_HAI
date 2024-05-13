import socket

SSI_BASE = 'SSI'
START_BASE = 'STRT'
STOP_BASE = 'STOP'
HANDSHAKE_MESSAGE = 'SSI:STRT:RUN1\0'
CLOSING_MESSAGE = 'SSI:STOP:RUN1\0'

def create_socket(host, port, task="send"):
    '''
    Create and connect to a socket.

    Args:
        host (str): The host to connect to
        port (int): The port to connect to
        task (string): create a socket to send data ("send") or receive them ("receive")
    
    Returns:
        socket (socket.socket,None): socket object
    '''

    client_socket = None
    if task == "send":
        # create a socket and send data to the server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_socket.sendto(bytes(HANDSHAKE_MESSAGE, "utf-8"), (host, port))
        print("Socket connected")
        return client_socket
    elif task == "receive":
        # create a socket and receive data from the server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_socket.bind((host, port))
        print("Socket connected")
        return client_socket

def close_socket(client_socket: socket, host, port):
    '''
    Send a closing message and close socket connection.
    
    Args:
        client_socket (socket.socket): socket object
        host (str): The host to connect to
        port (int): The port to connect to
    '''
    
    client_socket.sendto(bytes(CLOSING_MESSAGE, "utf-8"), (host, port))
    client_socket.close()
    print("Socket closed")

def send_data(client_socket: socket, host, port, data):
    '''
    Send data via socket.
    Args:
        client_socket (socket.socket): socket object
        host (str): The host to connect to
        port (int): The port to connect to
        data (string): Data to send
    '''

    client_socket.sendto(bytes(data + "\0", "utf-8"), (host, port))

def receive_data(client_socket: socket):
    '''
    Receive data via socket.
    
    Args:
        client_socket (socket.socket): socket object

    Returns:
        string: Data received
    '''

    data, addr = client_socket.recvfrom(1024)
    return data.decode("utf-8")