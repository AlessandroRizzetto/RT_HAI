#0.8898800601115412

import json
import keyboard
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
    def __init__(self, max_values, ser=None) -> None:
        self.actual_pattern = 0
        self.actual_min_intensity = 0
        self.actual_max_intensity = 0
        self.actual_pace = 0
        self.ser = ser
        self.max_values = max_values
        self.values = []

    def add_data(self, new_data):
        self.values.append(new_data)
        if len(self.values) > self.max_values:
            self.values.pop(0)
    
    def send_update(self, code, reference, pattern, min_intensity, max_intensity, pace):
        seh.send_data(self.ser, f"<{code},{reference},{pattern},{min_intensity},{max_intensity},{pace}>")
        self.actual_pattern = pattern
        self.actual_min_intensity = min_intensity
        self.actual_max_intensity = max_intensity
        self.actual_pace = pace

class SerialVideoFeature(SerialFeature):
    def __init__(self, max_values, ser=None) -> None:
        super().__init__(max_values, ser)
        

    def convert_and_send_update(self, new_message, new_value = None):
        correct_update = False
        if new_message == "HANDS_NOT_VISIBILITY":
            self.send_update("v", 4, 1, 30, 100, 50) # sinusoide
            correct_update = True
            print("Message sent to the arduino, hands not visible")
        if new_message == "HANDS_VISIBILITY":
            self.send_update("v", 4, 0, 0, 0, 0)
            correct_update = True
            print("Message sent to the arduino, hands visible")

        if new_message == "HANDS_TOUCHING":
            self.send_update("v", 4, 0, 40, 40, 0)
            correct_update = True
            print("Message sent to the arduino, hands touching")
        if new_message == "HANDS_NOT_TOUCHING":
            self.send_update("v", 4, 0, 0, 0, 0)
            correct_update = True
            print("Message sent to the arduino, hands not touching")

        if new_message == "BAD_BODY_DIRECTION":
            if new_value == "Body Right":
                self.send_update("v", 3, 0, 30, 30, 0)
                correct_update = True
                print("Message sent to the arduino, body right")
            elif new_value == "Body Left":
                self.send_update("v", 2, 0, 30, 30, 0)
                correct_update = True
                print("Message sent to the arduino, body left")
        if new_message == "GOOD_BODY_DIRECTION":
            self.send_update("v", 5, 0, 0, 0, 0)
            correct_update = True
            print("Message sent to the arduino, body forward")

        # if new_message == "CROUCH":
        #     self.send_update("v", 5, 1, 20, 40, 0)
        #     correct_update = True
        #     print("Message sent to the arduino, crouch")
        # if new_message == "NOT_CROUCH":
        #     self.send_update("v", 5, 0, 0, 0, 0)
        #     correct_update = True
        #     print("Message sent to the arduino, not crouch")

        if new_message == "BAD_HEAD_DIRECTION":
            self.send_update("h", 0, 0, 10, 10, 5)
            correct_update = True
            print("Message sent to the arduino, head not forward")
        if new_message == "GOOD_HEAD_DIRECTION":
            self.send_update("h", 0, 0, 0, 0, 10)
            correct_update = True
            print("Message sent to the arduino, head forward")
        
        if correct_update:
            self.add_data({
                "message": new_message,
                "value": new_value,
            })

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

def setup_plots():
    global arrows_plot, lines_plot

    fig, axs = plt.subplots(2, 2, sharex='col')
    
    axs[0, 0].set_title(arrows_plot["loudness"].title)
    axs[0, 0].set_xlim(0, 0.5)
    axs[0, 0].set_autoscale_on(True)
    axs[1, 0].set_title(arrows_plot["pitch"].title)
    axs[1, 0].set_xlim(0, 0.5)
    axs[1, 0].set_autoscale_on(True)
    
    axs[0, 1].set_title(lines_plot["loudness-d"].title)
    axs[1, 1].set_title(lines_plot["pitch-d"].title)

    plt.show()
    plt.pause(0.25)

    return fig, axs, arrows_plot, lines_plot

def stop_execution():
    global loop, sock, HOST, PORT
    loop = False
    soh.close_socket(sock, HOST, PORT)

HOST = '127.0.0.1' # The server's hostname or IP address
PORT = 4444       # The port used by the server

plt.ion()

if __name__ == "__main__":

    feedback_audio = {
        "visual": False,
        "aptic": False,
    }
    feedback_video = {
        "aptic": False,
    }

    arrows_plot = {
        "loudness": ArrowPlotInformation([0, 0], "Loudness derivative"),
        "pitch": ArrowPlotInformation([1, 0], "Pitch derivative")
    }
    lines_plot = {
        "loudness-d": LinePlotInformation([0, 1], "Loudness derivative history", 20),
        "pitch-d": LinePlotInformation([1, 1], "Pitch derivative history", 20)
    }
    serial_features = {
        "loudness": SerialFeature(5)
    }
    
    video_features = {
        "crouch": SerialVideoFeature(1),
        "hands_distance": SerialVideoFeature(1),
        "hands_visibility": SerialVideoFeature(1),
        "body_direction": SerialVideoFeature(1),
        "head_direction": SerialVideoFeature(1),
    }

    user_class = "nd"
    ser = None
    fig = None
    axs = None

    loop = True
    keyboard.add_hotkey('esc', stop_execution)

    # Create a socket object
    sock = soh.create_socket(HOST, PORT, "receive")
    sock.settimeout(30)

    user_class, neutral_features = get_user_neutral_features()

    while loop:
        # Receive data from the server
        try:
            data = soh.receive_data(sock)
        except:
            data = ""
        
        if data == soh.HANDSHAKE_MESSAGE:
            print("Received handshake message")
        elif data == soh.CLOSING_MESSAGE:
            print("Received closing message")
        elif data == "":
            pass
        elif data == f"{soh.SSI_BASE}:{soh.START_BASE}:AUDIO_VISUAL\0":
            print("Activating audio visual feedback")
            if fig is None:
                fig, axs, arrows_plot, lines_plot = setup_plots()
            feedback_audio["visual"] = True
        elif data == f"{soh.SSI_BASE}:{soh.STOP_BASE}:AUDIO_VISUAL\0":
            print("Stopping audio visual feedback")
            if fig is not None:
                plt.close(fig)
                plt.pause(1)
                fig = None
                axs = None
            feedback_audio["visual"] = False
        elif data == f"{soh.SSI_BASE}:{soh.START_BASE}:AUDIO_APTIC\0":
            print("Activating audio aptic feedback")
            if ser is None:
                ser = seh.connect_serial('COM5', 9600)
                for key in serial_features:
                    serial_features[key].ser = ser
            feedback_audio["aptic"] = True
        elif data == f"{soh.SSI_BASE}:{soh.STOP_BASE}:AUDIO_APTIC\0":
            print("Stopping audio aptic feedback")
            if ser is not None:
                seh.close_serial(ser)
                ser = None
                for key in serial_features:
                    serial_features[key].ser = None
            feedback_audio["visual"] = False
        elif data == f"{soh.SSI_BASE}:{soh.START_BASE}:VIDEO_APTIC\0":
            print("Activating video aptic feedback")
            if ser is None:
                ser = seh.connect_serial('COM5', 9600)
                for key in video_features:
                    video_features[key].ser = ser
            feedback_video["aptic"] = True
        elif data == f"{soh.SSI_BASE}:{soh.STOP_BASE}:VIDEO_APTIC\0":
            print("Stopping video aptic feedback")
            if ser is not None:
                seh.close_serial(ser)
                ser = None
                for key in video_features:
                    video_features[key].ser = None
            feedback_video["visual"] = False
        else:
            try:
                json_data = json.loads(data.split('\n')[0])
            except:
                print("Wrong data format:", repr(data))
                json_data = {}

            for key in json_data:
                if feedback_audio["visual"] and key in arrows_plot:
                    arrows_plot[key].update_plot(json_data[key][0])
                
                if feedback_audio["visual"] and key in lines_plot:
                    lines_plot[key].update_plot(json_data[key][0])
                
                if ser is not None:
                    if feedback_audio["aptic"] and key == "loudness":
                        serial_features[key].add_data(json_data[key][0])
                        print(len(serial_features[key].values), serial_features[key].max_values)
                        print(np.mean(serial_features[key].values),neutral_features["loudness"],np.mean(serial_features[key].values) - neutral_features["loudness"])
                        if len(serial_features[key].values) == serial_features[key].max_values:
                            difference = np.mean(serial_features[key].values) - neutral_features["loudness"]
                            if serial_features[key].actual_max_intensity == 0 and difference > 0.1:
                                serial_features[key].send_update("h", 1, 0, 30, 30, 10)
                                print("Message sent to the arduino, level of voice too loud")
                            elif serial_features[key].actual_max_intensity > 0 and difference <= 0.1:
                                serial_features[key].send_update("h", 1, 0, 0, 0, 10)
                                print("Message sent to the arduino, level of voice correct")
                        
                    if feedback_video["aptic"] and key in video_features:
                        new_message = json_data[key]["message"] if "message" in json_data[key] else ""
                        new_value = json_data[key]["value"] if "value" in json_data[key] else None
                        video_features[key].convert_and_send_update(new_message, new_value)
                        
                if feedback_audio["visual"]:
                    plt.pause(0.1)
    
    print("Feedback Execution stopped")