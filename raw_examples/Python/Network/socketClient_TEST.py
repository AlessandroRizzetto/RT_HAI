
import socket
import time

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 2222      # The port used by the server
SYNC = 1111
MESSAGE = 'SSI:STRT:RUN1\0'

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    # s.connect((HOST, SYNC))

    s.sendto(bytes(MESSAGE, "utf-8"), (HOST, SYNC))
    i = 0
    # s.close()

    # s.connect((HOST, PORT))
    while True:
        # send udp packet
        s.sendto(bytes(str(i) + "\0", "utf-8"), (HOST, PORT))
        print('sent', i)
        i += 1
        time.sleep(0.1)

# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket:
#     socket.connect((HOST, PORT))
#     i = 0
#     while True:
#         socket.sendall(bytes(str(i), "utf-8"))
#         print('sent', i)
#         i += 1
#         time.sleep(1)

# import socket

# UDP_IP = "127.0.0.1"
# UDP_PORT = 1111
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# input("\n\tpress enter to start pipeline\n")

# MESSAGE = "SSI:STRT:RUN1\0"
# sock.sendto(bytes(MESSAGE, "utf-8"), (UDP_IP, UDP_PORT))

# input("\n\tpress enter to stop pipeline\n")

# MESSAGE = "SSI:STOP:RUN1\0"
# sock.sendto(bytes(MESSAGE, "utf-8"), (UDP_IP, UDP_PORT))
