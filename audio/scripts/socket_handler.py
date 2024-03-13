import socket

HANDSHAKE_MESSAGE = 'SSI:STRT:RUN1\0'

def manage_socket(host, port, state, task="send"):
    if state == "start":
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
    if state == "stop":
        print("Socket closed")
        client_socket.close()
        return None

def send_data_network(client_socket: socket, host, sender, data):
    client_socket.sendto(bytes(data + "\0", "utf-8"), (host, sender))