import numpy as np

def transform(info, sin, sout, sxtras, board, opts, vars):   
    npin = np.asmatrix(sin)
    npout = np.asmatrix(sout)
    
    np.mean(np.abs(npin)) / np.max(np.abs(npin), out=npout)