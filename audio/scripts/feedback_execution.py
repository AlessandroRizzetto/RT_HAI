import json
import serial_handler as seh
import socket_handler as soh
import matplotlib.pyplot as plt

class PlotInformation:
    def __init__(self, plot_position, title):
        self.plot_position = plot_position
        self.title = title

class LinePlotInformation(PlotInformation):
    def __init__(self, plot_position, title, max_values):
        super().__init__(plot_position, title)
        self.max_values = max_values
        self.x = []
        self.y = []

    def update_plot(self, new_value):
        global axs
        
        if len(self.x) < self.max_values:
            self.x.append(self.x[-1] + 1 if len(self.x) > 0 else 0)
        self.y.insert(0,new_value)
        if len(self.y) > self.max_values:
            self.y.pop()
    
        # update the correspodent plot in axs
        axs[self.plot_position[0]][self.plot_position[1]].clear()
        axs[self.plot_position[0]][self.plot_position[1]].plot(self.x, self.y)
        axs[self.plot_position[0]][self.plot_position[1]].set_xlim(self.x[0], self.x[-1])
        axs[self.plot_position[0]][self.plot_position[1]].relim()
        axs[self.plot_position[0]][self.plot_position[1]].autoscale_view(True,True,True)
        '''ylim = axs[self.plot_position[0]][self.plot_position[1]].get_ylim()
        if new_value != ylim[0] and new_value != ylim[1]:
            axs[self.plot_position[0]][self.plot_position[1]].set_ylim(ymin=min(new_value,ylim[0]), ymax=max(new_value,ylim[1]))
        print(ylim, new_value, axs[self.plot_position[0]][self.plot_position[1]].get_ylim())'''
        plt.draw()

class ArrowPlotInformation(PlotInformation):
    def __init__(self, plot_position, title):
        super().__init__(plot_position, title)
        self.value = 0

    def update_plot(self, new_value):
        global axs

        direction = 1 if new_value > self.value else -1
        self.value = new_value
        axs[self.plot_position[0]][self.plot_position[1]].clear()
        axs[self.plot_position[0]][self.plot_position[1]].arrow(0.25, 0, 0, direction*1, width=0.025, head_width=0.05, head_length=0.85, fc='k', ec='k')
        plt.draw()

HOST = '127.0.0.1' # The server's hostname or IP address
PORT = 4444       # The port used by the server

plt.ion()

if __name__ == "__main__":
    # Create a socket object
    sock = soh.create_socket(HOST, PORT, "receive")

    sock.settimeout(15)

    fig, axs = plt.subplots(2, 2, sharex='col')

    arrows_plot = {
        "loudness": ArrowPlotInformation([0, 0], "Loudness derivative"),
        "pitch": ArrowPlotInformation([1, 0], "Pitch derivative")
    }
    axs[0, 0].set_title(arrows_plot["loudness"].title)
    axs[0, 0].set_xlim(0, 0.5)
    axs[0, 0].set_autoscale_on(True)
    axs[1, 0].set_title(arrows_plot["pitch"].title)
    axs[1, 0].set_xlim(0, 0.5)
    axs[1, 0].set_autoscale_on(True)

    lines_plot = {
        "loudness-d": LinePlotInformation([0, 1], "Loudness derivative history", 20),
        "pitch-d": LinePlotInformation([1, 1], "Pitch derivative history", 20)
    }
    axs[0, 1].set_title(lines_plot["loudness-d"].title)
    axs[1, 1].set_title(lines_plot["pitch-d"].title)

    plt.show()
    plt.pause(0.25)

    while True:
        # Receive data from the server
        print("Waiting for data")
        data = soh.receive_data(sock)
        print("Data received")
        
        if data == soh.HANDSHAKE_MESSAGE:
            print("Received handshake message")
        elif data == soh.CLOSING_MESSAGE:
            print("Received closing message")
            soh.close_socket(sock, HOST, PORT)
            break
        else:
            json_data = json.loads(data.split('\n')[0])

            for key in json_data:
                if key in arrows_plot:
                    arrows_plot[key].update_plot(json_data[key][0])
                
                if key in lines_plot:
                    lines_plot[key].update_plot(json_data[key][0])
                
                plt.pause(0.01)