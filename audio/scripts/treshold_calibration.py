import numpy as np
import sys
import socket_handler as sh

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.widgets import Button

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 3333      # The port used by the server

def check_trehshold(tresh):
    if init_tresh < 0 or init_tresh > 1:
        print('Invalid treshold value. Please enter a value between 0 and 1.')
        return False
    return True

# The function to be called anytime a slider's value changes
def update(val):
    sh.send_data(client_socket, HOST, PORT, str(tresh_slider.val) + "\n")
    bar[0].set_height(tresh_slider.val)
    fig.canvas.draw_idle()

def reset(event):
    tresh_slider.reset()

if __name__ == "__main__":
    # Define initial parameters
    default = True
    if len(sys.argv) > 1:
        try:
            init_tresh = float(sys.argv[1])
            if check_trehshold(init_tresh):
                default = False
        except ValueError:
            print('Invalid treshold value from command line')

    if len(sys.argv) == 0 or default:
        with open('../audio/pipes/vad_filter.pipeline-config', 'r') as file:
            old_configs = file.readlines()
            for line in old_configs:
                tmp_line = line.split(" = ")
                if 'vad:tresh' == tmp_line[0]:
                    try:
                        init_tresh = float(tmp_line[1])
                        if check_trehshold(init_tresh):
                            default = False
                            break
                    except ValueError:
                        print('Invalid treshold value from config')

    if default:
        init_tresh = 0.4
        print('Using default treshold:',init_tresh)

    # Create the figure and the line that we will manipulate
    mpl.rcParams['toolbar'] = 'None'
    fig, ax = plt.subplots(figsize=(3, 5))
    x = ['']
    y = [init_tresh]

    bar = ax.bar(x, y)
    ax.set_title('Treshold')
    ax.set_ylim(0, 1)

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.35, right=0.65, bottom=0.25)

    # Make a horizontal slider to control the treshold
    axtresh = fig.add_axes([0.25, 0.1, 0.6, 0.03])
    tresh_slider = Slider(
        ax=axtresh,
        label='',
        valmin=0,
        valmax=1.0,
        valinit=init_tresh,
    )

    # register the update function for the slider
    tresh_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    # Initialize the client_socket variable
    client_socket = None

    resetax = fig.add_axes([0.05, 0.1, 0.15, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    button.on_clicked(reset)

    # Set the window position
    mngr = plt.get_current_fig_manager()
    try:
        mngr.window.wm_geometry("250x350+300+400")
    except:
        pass

    # Start the socket
    client_socket = sh.create_socket(HOST, PORT, "send")
    if client_socket is None:
        print("Error starting the socket")
        sys.exit(1)
    else:
        # Show the plot
        plt.show()

        # Close the socket
        sh.close_socket(client_socket, HOST, PORT)