import os
import csv
import numpy as np
from scipy.signal import savgol_filter

def getOptions(opts,vars):
    opts['new_file'] = 'false'
    opts['file_path'] = '../data/dataTable.csv'
    opts['features'] = ['loudness', 'loudness-d', 'pitch', 'pitch-d', 'energy', 'jitter', 'shimmer', 'alpha-ratio', 'hammarberg-index', 'spectral-flux', 'spectral-slope']
    opts['user_class'] = 'CLASSE'
    opts['mean'] = 'false'
    opts['polyorder'] = 2

    vars['data'] = {opts['features'][i]: np.array([]) for i in range(len(opts['features']))}

def consume_enter(sins, board, opts, vars):
    '''
    SSI function called before the beginning of the processing of the consumer.
    Open file for writing. If file exists and is not requested a new file, append to it. If not, create a new file and write the header.
    
    Args:
        sins (tuple<ssipystream>): list of signals
        board (ssipyeventboard): event board to send events
        opts (dictionary<string,any>): options of the component
        vars (dictionary<string,any>): internal variables of the component
    '''
    new_file = os.path.exists(opts['file_path']) and opts['new_file'] == 'true'
    try:
        vars['f'] = open(opts['file_path'], 'w') if new_file else open(opts['file_path'], 'a')
    except:
        print("Error opening file")
        vars['f'] = None
    
    if vars['f'] is not None:
        vars['writer'] = csv.writer(vars['f'], delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        if new_file:
            first_row = ['class']
            first_row.extend(opts['features'])
            vars['writer'].writerow(first_row)
    else:
        vars['writer'] = None

def consume(info, sin, board, opts, vars):
    '''
    SSI function called when new samples are received.
    Append the new samples to the data dictionary and save them on file if specified.
    '''
    to_write = [opts['user_class']]
    for i, x in enumerate(sin):
        npin = np.array(x)
        npin = npin.flatten()

        # Clean samples
        # Remove zeros
        if npin.size != 0:
            npin = npin[npin != 0.0]
        # Apply Savitzky-Golay filter
        if npin.size > opts['polyorder']:
            npin = savgol_filter(npin, npin.size, opts['polyorder'])

        # Append to data dictionary
        vars['data'][opts['features'][i]] = np.concatenate((vars['data'][opts['features'][i]], npin))
        
        # Write to file the samples
        if opts['mean'] == 'false' and vars['writer'] is not None:
            if npin.size > 1:
                to_write.append(np.mean(npin))
            elif npin.size == 1:
                to_write.append(npin[0])
            else:
                to_write.append(0)
    if opts['mean'] == 'false' and vars['writer'] is not None:
        vars['writer'].writerow(to_write)
        

def consume_flush(sins, board, opts, vars):
    '''
    SSI function called when the processing was stopped.
    Calculate the mean of the samples and write them to the file if specified.

    Args:
        sins (tuple<ssipystream>): list of signals
        board (ssipyeventboard): event board to send events
        opts (dictionary<string,any>): options of the component
        vars (dictionary<string,any>): internal variables of the component
    '''
    to_write = [opts['user_class']]
    for x in vars['data']:
        if vars['data'][x].size > 0:
            if vars['data'][x].size > opts['polyorder']:
                vars['data'][x] = savgol_filter(vars['data'][x], vars['data'][x].size, opts['polyorder'])
            to_write.append(np.mean(vars['data'][x]))
        else:
            to_write.append(0)
    
    if opts['mean'] == 'true' and vars['writer'] is not None:
        vars['writer'].writerow(to_write)
    else:
        print("Final means:")
        for i, x in enumerate(vars['data']):
            print(x, to_write[i+1])
    
    if vars['f'] is not None:
        vars['f'].close()