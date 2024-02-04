import sys
import numpy as np
import fft_tools as ft

def getOptions(opts, vars):
   opts['eps'] = sys.float_info.epsilon
   vars['last_fft_magnitude'] = None

def transform(info, sin, sout, sxtras, board, opts, vars):   
    npin = np.asarray(sin)#np.asmatrix(sin)
    
    if vars['last_fft_magnitude'] is None:
        vars['last_fft_magnitude'] = ft.fft_magnitude(npin)
        sout[0] = 0
    else:
        tmp_fft = ft.fft_magnitude(npin)
        sout[0] = spectral_flux(tmp_fft, vars['last_fft_magnitude'], opts['eps'])
        vars['last_fft_magnitude'] = tmp_fft

def spectral_flux(fft_magnitude, previous_fft_magnitude, eps):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        fft_magnitude:            the abs(fft) of the current frame
        previous_fft_magnitude:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    fft_sum = np.sum(fft_magnitude + eps)
    previous_fft_sum = np.sum(previous_fft_magnitude + eps)
    sp_flux = np.sum(
        (fft_magnitude / fft_sum - previous_fft_magnitude /
         previous_fft_sum) ** 2)

    return sp_flux