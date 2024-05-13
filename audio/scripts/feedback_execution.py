import json
import keyboard
import pandas as pd
import serial_handler as seh
import socket_handler as soh
import matplotlib.pyplot as plt
import numpy as np

class PlotInformation:
    '''
    Abstract common class to manage plots.

    Attributes:
        plot_position (int[]): list with [r,c] to indicate the plot position in the figure grid
        plot_position (string): plot title
    '''

    def __init__(self, plot_position, title):
        '''
        Plot init.
        
        Args:
            plot_position (int[]): list with [r,c] to indicate the plot position in the figure grid
            plot_position (string): plot title
        '''

        self.plot_position = plot_position
        self.title = title

class LinePlotInformation(PlotInformation):
    '''
    Class to manage line plots. Extention of Plot information.

    Additional attributes:
        max_values (int): maximum number of value to show in the plot
        x (float[]): plot's x values
        y (float[]): plot's y values
    '''

    def __init__(self, plot_position, title, max_values):
        '''
        Line plot init.
        
        Args:
            plot_position (int[]): list with [r,c] to indicate the plot position in the figure grid
            plot_position (string): plot title
            max_values (int): maximum number of value to show in the plot
        '''

        super().__init__(plot_position, title)
        self.max_values = max_values
        self.x = []
        self.y = []

    def update_plot(self, new_value):
        '''
        Update plot adding a new value. The oldest value is deleted.
        
        Args:
            new_value (int): new value to add
        '''

        global axs
        
        # Add new value and delete the oldest, if there are more than max_values
        if len(self.x) < self.max_values:
            self.x.append(self.x[-1] + 1 if len(self.x) > 0 else 0)
        self.y.insert(0,new_value)
        if len(self.y) > self.max_values:
            self.y.pop()
    
        # update the correspodent plot in axs
        tmp_title = axs[self.plot_position[0]][self.plot_position[1]].get_title()
        axs[self.plot_position[0]][self.plot_position[1]].clear()
        axs[self.plot_position[0]][self.plot_position[1]].plot(self.x, self.y)
        axs[self.plot_position[0]][self.plot_position[1]].relim()
        axs[self.plot_position[0]][self.plot_position[1]].autoscale_view(True,True,True)
        axs[self.plot_position[0]][self.plot_position[1]].set_title(tmp_title)
        '''ylim = axs[self.plot_position[0]][self.plot_position[1]].get_ylim()
        if new_value != ylim[0] and new_value != ylim[1]:
            axs[self.plot_position[0]][self.plot_position[1]].set_ylim(ymin=min(new_value,ylim[0]), ymax=max(new_value,ylim[1]))
        print(ylim, new_value, axs[self.plot_position[0]][self.plot_position[1]].get_ylim())'''
        plt.draw()

class ArrowPlotInformation(PlotInformation):
    '''
    Class to manage arrow plot (only y value of the arrow).
    
    Additional attributes:
        value (int): Plot value (i.e., arrow's y value)
    '''

    def __init__(self, plot_position, title):
        '''
        Plot init.
        
        Args:
            plot_position (int[]): list with [r,c] to indicate the plot position in the figure grid
            plot_position (string): plot title
        '''

        super().__init__(plot_position, title)
        self.value = 0

    def update_plot(self, new_value):
        '''
        Update plot changing changing the y value of the arrow (mainly and eventually changing its direction).
        
        Args:
            new_value (int): new arrow y balue
        '''

        global axs

        direction = 1 if new_value > self.value else -1
        self.value = new_value
        tmp_title = axs[self.plot_position[0]][self.plot_position[1]].get_title()
        axs[self.plot_position[0]][self.plot_position[1]].clear()
        axs[self.plot_position[0]][self.plot_position[1]].arrow(0.25, 0, 0, direction*1, width=0.025, head_width=0.05, head_length=0.85, fc='k', ec='k')
        axs[self.plot_position[0]][self.plot_position[1]].set_title(tmp_title)
        plt.draw()

class SerialFeature:
    '''
    Class to manage features to serially send.

    Attributes:
        actual_pattern (int): last pattern sent to Arduino
        actual_min_intensity (int): lats min_intensity sent
        actual_max_intensity (int): last max_intensity sent
        actual_pace (int): last pace sent
        ser = ser (serial.Serial): port to send messages
        max_values (int): max number of values to store
        values (float): values stored
    '''

    def __init__(self, max_values, ser=None) -> None:
        '''
        Plot init.
        
        Args:
            max_values (int): max number of values to store
            ser (serial.Serial): port to send messages
        '''

        self.actual_pattern = 0
        self.actual_min_intensity = 0
        self.actual_max_intensity = 0
        self.actual_pace = 0
        self.ser = ser
        self.max_values = max_values
        self.values = []

    def add_data(self, new_data):
        '''
        Adding a new value. The oldest value is deleted.
        
        Args:
            new_value (int): new value to add
        '''

        self.values.append(new_data)
        if len(self.values) > self.max_values:
            self.values.pop(0)
    
    def send_update(self, code, reference, pattern, min_intensity, max_intensity, pace):
        '''
        Adding a new value. The oldest value is deleted.
        
        Args:
            new_value (int): new value to add
        '''

        seh.send_data(self.ser, f"<{code},{reference},{pattern},{min_intensity},{max_intensity},{pace}>")
        self.actual_pattern = pattern
        self.actual_min_intensity = min_intensity
        self.actual_max_intensity = max_intensity
        self.actual_pace = pace

class SerialVideoFeature(SerialFeature):
    '''
    Class which extendeds SerialFeatures and manage video features.
    '''

    def __init__(self, max_values, ser=None) -> None:
        '''
        Plot init.
        
        Args:
            max_values (int): max number of values to store
            ser (serial.Serial): port to send messages
        '''

        super().__init__(max_values, ser)
        

    def convert_and_send_update(self, new_message, new_value = None):
        '''
        Convert video features into Arduino messages, following the defined protocol.
        
        Args:
            new_message (string): new video feature
            new_value (boolean): specify if new_message is different from the previous
        '''

        correct_update = False
        if new_message == "HANDS_NOT_VISIBILITY":
            self.send_update("v", 4, 1, 30, 100, 50) # sinusoid
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

        if new_message == "CROUCH":
            self.send_update("v", 5, 1, 20, 40, 0)
            correct_update = True
            print("Message sent to the arduino, crouch")
        if new_message == "NOT_CROUCH":
            self.send_update("v", 5, 0, 0, 0, 0)
            correct_update = True
            print("Message sent to the arduino, not crouch")

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
    '''
    Get user_class neutral features form file.

    Returns:
        user_class (str): user class
        neutral_features (dict): neutral features
    '''

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
    '''
    Setup figure and subplots within as specified.

    Returns:
        fig (matplotlib.figure): plot figure
        axs (matplotlib.axes.Axes): grid of subplots
        arrows_plot (dict<string,ArrowPlotInformation>): dictionary which contains arrows plots associated to their features
        lines_plot (dict<string,LinePlotInformation>): dictionary which contains line plots associated to their features
    '''

    global arrows_plot, lines_plot

    fig, axs = plt.subplots(2, 2, sharex='col', figsize=(7, 6))
    
    axs[0, 0].set_xlim(0, 0.5)
    axs[0, 0].xaxis.set_ticks([])
    axs[0, 0].set_autoscale_on(True)
    axs[1, 0].set_xlim(0, 0.5)
    axs[0, 0].xaxis.set_ticks([])
    axs[1, 0].set_autoscale_on(True)
    
    axs[0, 0].set_title(arrows_plot["loudness"].title)
    axs[1, 0].set_title(arrows_plot["pitch"].title)
    axs[0, 1].set_title(lines_plot["loudness-d"].title)
    axs[1, 1].set_title(lines_plot["pitch-d"].title)

    plt.show()
    plt.pause(0.25)

    return fig, axs, arrows_plot, lines_plot

def stop_execution():
    '''
    Stops the execution of the program.
    '''

    global loop, sock, HOST, PORT
    
    loop = False
    soh.close_socket(sock, HOST, PORT)

HOST = '127.0.0.1' # The server's hostname or IP address
PORT = 4444       # The port used by the server
SERIAL_PORT = 'COM5' # Port used by Arduino

plt.ion()

if __name__ == "__main__":

    #Setup feedbacks and plots
    audio_feedbacks = {
        "visual": False,
        "haptic": False,
    }
    video_feedback = {
        "haptic": False,
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

    # Initialize loop and possible interruption ("esc" key on keyboard)
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
            audio_feedbacks["visual"] = True
        elif data == f"{soh.SSI_BASE}:{soh.STOP_BASE}:AUDIO_VISUAL\0":
            print("Stopping audio visual feedback")
            if fig is not None:
                plt.close(fig)
                plt.pause(1)
                fig = None
                axs = None
            audio_feedbacks["visual"] = False
        elif data == f"{soh.SSI_BASE}:{soh.START_BASE}:AUDIO_APTIC\0":
            print("Activating audio haptic feedback")
            if ser is None:
                ser = seh.connect_serial(SERIAL_PORT, 9600)
                for key in serial_features:
                    serial_features[key].ser = ser
            audio_feedbacks["haptic"] = True
        elif data == f"{soh.SSI_BASE}:{soh.STOP_BASE}:AUDIO_APTIC\0":
            print("Stopping audio haptic feedback")
            if ser is not None:
                seh.close_serial(ser)
                ser = None
                for key in serial_features:
                    serial_features[key].ser = None
            audio_feedbacks["visual"] = False
        elif data == f"{soh.SSI_BASE}:{soh.START_BASE}:VIDEO_APTIC\0":
            print("Activating video haptic feedback")
            if ser is None:
                ser = seh.connect_serial(SERIAL_PORT, 9600)
                for key in video_features:
                    video_features[key].ser = ser
            video_feedback["haptic"] = True
        elif data == f"{soh.SSI_BASE}:{soh.STOP_BASE}:VIDEO_APTIC\0":
            print("Stopping video haptic feedback")
            if ser is not None:
                seh.close_serial(ser)
                ser = None
                for key in video_features:
                    video_features[key].ser = None
            video_feedback["visual"] = False
        else:
            try:
                json_data = json.loads(data.split('\n')[0])
            except:
                print("Wrong data format:", repr(data))
                json_data = {}

            # Execute feedbacks and update plots specified in the body message
            for key in json_data:
                if audio_feedbacks["visual"] and key in arrows_plot:
                    arrows_plot[key].update_plot(json_data[key][0])
                
                if audio_feedbacks["visual"] and key in lines_plot:
                    lines_plot[key].update_plot(json_data[key][0])
                
                if ser is not None:
                    if audio_feedbacks["haptic"] and key == "loudness":
                        serial_features[key].add_data(json_data[key][0])
                        if len(serial_features[key].values) == serial_features[key].max_values:
                            difference = np.mean(serial_features[key].values) - neutral_features["loudness"]
                            if serial_features[key].actual_max_intensity == 0 and difference > 0.1: # If loudness increased enough activate proper feedback
                                serial_features[key].send_update("h", 1, 0, 30, 30, 10)
                                print("Message sent to the arduino, level of voice too loud")
                            elif serial_features[key].actual_max_intensity > 0 and difference <= 0.1: # If loudness increased enough disable proper feedback
                                serial_features[key].send_update("h", 1, 0, 0, 0, 10)
                                print("Message sent to the arduino, level of voice correct")
                        
                    if video_feedback["haptic"] and key in video_features:
                        new_message = json_data[key]["message"] if "message" in json_data[key] else ""
                        new_value = json_data[key]["value"] if "value" in json_data[key] else None
                        video_features[key].convert_and_send_update(new_message, new_value)
                        
            if audio_feedbacks["visual"]:
                plt.pause(0.1)
    
    print("Feedback Execution stopped")