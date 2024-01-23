import numpy as np
import hammarberg_index as hi

def transform(info, sin, sout, sxtras, board, opts, vars):   
    npin = np.asarray(sin)
    npin = npin.reshape(-1) if npin.shape[1] == 1 else npin
    
    sout[0] = hi.hammIndex(sin.sr, npin)[0]