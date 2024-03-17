import json
import pandas as pd
import serial_handler as seh
import socket_handler as soh
import matplotlib.pyplot as plt
import numpy as np

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

class SerialFeature:
    def __init__(self, code, reference, max_values) -> None:
        self.code = code
        self.reference = reference
        self.actual_pattern = 0
        self.actual_min_intensity = 0
        self.actual_max_intensity = 0
        self.actual_pace = 0
        self.max_values = max_values
        self.values = []

    def add_data(self, new_data):
        self.values.append(new_data)
        if len(self.values) > self.max_values:
            self.values.pop(0)
    
    def send_update(self, pattern, min_intensity, max_intensity, pace):
        seh.send_data(f"<{self.code},{self.reference},{pattern},{min_intensity},{max_intensity},{pace}>")
        self.actual_pattern = pattern
        self.actual_min_intensity = min_intensity
        self.actual_max_intensity = max_intensity
        self.actual_pace = pace

def get_user_neutral_features():
    user_class = "nd"
    with open('../audio/pipes/audio_features.pipeline-config', 'r') as file:
        configs = file.readlines()
        for line in configs:
            line = line[:-1]
            tmp_line = line.split(" = ")
            if 'user:class' == tmp_line[0]:
                user_class = tmp_line[1]

    neutral_features = pd.read_csv('../audio/data/userCalibration.csv', index_col=0).loc[user_class].to_dict()
    
    return user_class, neutral_features

HOST = '127.0.0.1' # The server's hostname or IP address
PORT = 4444       # The port used by the server

plt.ion()

if __name__ == "__main__":
    # Create a socket object
    sock = soh.create_socket(HOST, PORT, "receive")
    sock.settimeout(30)

    user_class, neutral_features = get_user_neutral_features()
    print(user_class, neutral_features)

    ser = seh.connect_serial('COM5', 9600)
    serial_features = {
        "loudness": SerialFeature("h", 1, 5)
    }

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
        data = soh.receive_data(sock)
        
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
                
                if ser is not None:
                    if key == "loudness":
                        serial_features[key].add_data(json_data[key][0])
                        print("Len", len(serial_features[key].values), serial_features[key].max_values)
                        print("Diff", np.mean(serial_features[key].values), neutral_features["loudness"], np.mean(serial_features[key].values) - neutral_features["loudness"])
                        if len(serial_features[key].values) == serial_features[key].max_values:
                            difference = np.mean(serial_features[key].values) - neutral_features["loudness"]
                            if serial_features[key].actual_max_intensity == 0 and difference > 0.1:
                                serial_features[key].send_update(0, 30, 30, 5)
                                print("Message sent to the arduino, level of voice too loud")
                            elif serial_features[key].actual_max_intensity > 0 and difference <= 0.1:
                                serial_features[key].send_update(0, 0, 0, 5)
                                print("Message sent to the arduino, level of voice correct")
                        
                plt.pause(0.01)