import socket

HANDSHAKE_MESSAGE = 'SSI:STRT:RUN1\0'
CLOSING_MESSAGE = 'SSI:STOP:RUN1\0'

def create_socket(host, port, task="send"):
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
    client_socket.sendto(bytes(CLOSING_MESSAGE, "utf-8"), (host, port))
    client_socket.close()
    print("Socket closed")

def send_data(client_socket: socket, host, sender, data):
    client_socket.sendto(bytes(data + "\0", "utf-8"), (host, sender))

def receive_data(client_socket: socket):
    data, addr = client_socket.recvfrom(1024)
    return data.decode("utf-8")