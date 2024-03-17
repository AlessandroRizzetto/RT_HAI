import numpy as np

def getSampleDimensionOut(dim, opts, vars):
    return 1

def transform(info, sin, sout, sxtras, board, opts, vars):
    npin = np.asarray(sin)
    npin = npin.reshape(-1) if npin.shape[1] == 1 else npin

    sout[0] = np.diff(npin)