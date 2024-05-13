import numpy as np

def getSampleDimensionOut(dim, opts, vars):
    '''
    SSI function called to determine the number of dimensions in the output stream.

    Args:
        dim (int): output stream dimension
        opts (dictionary<string,any>): options of the component
        vars (dictionary<string,any>): internal variables of the component
    '''

    return 1

def transform(info, sin, sout, sxtras, board, opts, vars):
    '''
    SSI function called to transform the input samples into the output samples.
    Transform the input signal into its first derivative.

    Args:
        info (ssipyinfo): information about the input samples
        sin (ssipystream): input samples
        sout (ssipystream): output samples
        sextras (tuple<ssipystream>): extra streams of samples
        board (ssipyeventboard): event board to send events
        opts (dictionary<string,any>): options of the component
        vars (dictionary<string,any>): internal variables of the component
    '''
    
    npin = np.asarray(sin)
    npin = npin.reshape(-1) if npin.shape[1] == 1 else npin

    sout[0] = np.diff(npin)