import os
import csv
import numpy as np
from scipy.signal import savgol_filter

def getOptions(opts,vars):
    opts['file_path'] = '../data/dataTable.csv'
    opts['features'] = ['loudness', 'pitch', 'jitter', 'shimmer', 'energy', 'alpha-ratio', 'hammarberg-index', 'spectral-flux', 'spectral-slope']
    opts['user_class'] = 'CLASSE'
    opts['polyorder'] = 2

def consume_enter(sins, board, opts, vars):
    exists = os.path.exists('dataTable.csv')
    vars['f'] = open(opts['file_path'], 'w') if not exists else open(opts['file_path'], 'a')
    vars['writer'] = csv.writer(vars['f'], delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    if not exists:
        first_row = ['class']
        first_row.extend(opts['features'])
        vars['writer'].writerow(first_row)

def consume(info, sin, board, opts, vars):
    to_write = [opts['user_class']]
    for i in sin:
        npin = np.array(i)
        if npin.shape[0] != 0:
            npin = npin[npin != 0.0]
        if npin.shape[0] > opts['polyorder']:
            npin = savgol_filter(npin, npin.shape[0], opts['polyorder'])
        if npin.shape[0] > 1:
            to_write.append(np.mean(npin))
        elif npin.shape[0] == 1:
            to_write.append(npin[0])
        else:
            to_write.append(0)
    vars['writer'].writerow(to_write)
        

def consume_flush(sins, board, opts, vars):
    vars['f'].close()