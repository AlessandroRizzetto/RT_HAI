import json
import numpy as np
import socket_handler as sh

def getOptions(opts,vars):
    '''
    SSI function called to get the options of the component and initialize the variables.

    Args:
        opts (dictionary<string,any>): options of the component
        vars (dictionary<string,any>): internal variables of the component
    '''
    
    opts['features'] = ['loudness', 'loudness-d', 'pitch', 'pitch-d', 'energy', 'jitter', 'shimmer', 'alpha-ratio', 'hammarberg-index', 'spectral-flux', 'spectral-slope']
    opts['host'] = '127.0.0.1'  # The server's hostname or IP address
    opts['port'] = 4444      # The port used by the server
    opts['visual_feedback'] = True
    opts['aptic_feedback'] = True

def consume_enter(sins, board, opts, vars):
    '''
    SSI function called when the processing is started.
    Create the socket and start the connection to the server.
    
    Args:
        sins (tuple<ssipystream>): list of signals
        board (ssipyeventboard): event board to send events
        opts (dictionary<string,any>): options of the component
        vars (dictionary<string,any>): internal variables of the component
    '''

    vars['client_socket'] = sh.create_socket(opts['host'], opts['port'], "send")
    if vars['client_socket'] is None:
        print("Error starting the socket")
    else:
        if opts['visual_feedback']:
            sh.send_data(vars['client_socket'], opts['host'], opts['port'], f"{sh.SSI_BASE}:{sh.START_BASE}:AUDIO_VISUAL")
        if opts['aptic_feedback']:
            sh.send_data(vars['client_socket'], opts['host'], opts['port'], f"{sh.SSI_BASE}:{sh.START_BASE}:AUDIO_APTIC")
        print("Socket started")

def consume(info, sin, board, opts, vars):
    '''
    SSI function called when new samples are received.
    Send the samples to the server.
    '''

    to_send = {}
    for i, x in enumerate(sin):
        npin = np.array(x)
        npin = npin.flatten()

        # Append to dictionary
        to_send[opts['features'][i]] = npin.tolist()
    
    if vars['client_socket'] is not None:
        sh.send_data(vars['client_socket'], opts['host'], opts['port'], json.dumps(to_send) + "\n")

def consume_flush(sins, board, opts, vars):
    '''
    SSI function called when the processing was stopped.
    Close the socket.

    Args:
        sins (tuple<ssipystream>): list of signals
        board (ssipyeventboard): event board to send events
        opts (dictionary<string,any>): options of the component
        vars (dictionary<string,any>): internal variables of the component
    '''

    if vars['client_socket'] is not None:
        sh.send_data(vars['client_socket'], opts['host'], opts['port'], f"{sh.SSI_BASE}:{sh.STOP_BASE}:AUDIO_VISUAL")
        sh.send_data(vars['client_socket'], opts['host'], opts['port'], f"{sh.SSI_BASE}:{sh.STOP_BASE}:AUDIO_APTIC")
        sh.close_socket(vars['client_socket'], opts['host'], opts['port'])