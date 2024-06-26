# -*- coding: utf-8 -*-
"""
The following script is an implementation of the Hammarberg Index
following main specifications from the paper:
    HAMMARBERG, Britta, et al. Perceptual and acoustic correlates of abnormal voice qualities. 
    Acta oto-laryngologica, 1980, vol. 90, no 1-6, p. 441-451.

The INPUT signal must accomplish the following constraints:
    -Duration of at least 5 seconds
    -must be a noiseless speach audio signal    
    -Sampled by at least 12KHz 
    -format .wav
    
# ============================================================
    
EXAMPLE:    
    
    Index, fq, Pxx_dB = hammIndex(fs, x_norm)
    
INPUT:
    fs: Sampling rate in Hz
    x_norm: normalized signal [-1,1]         
      
OUTPUT:
    Index: Hammarberg Index in dB
    fq: frequency vector
    Pxx_dB: Power Spetral Density in dB     
        

# ============================================================
#
#  author: David Castro Piñol
#  email: davidpinyol91@gmail.com 
#
# ============================================================

Created on Wed Sep  8 11:09:35 2021


"""
from scipy import signal
import numpy as np
import sys


def hammIndex(fs, x):
    
    # edges frequencies     
    f1 = 2000
    f2 = 5000      
    
    # checking and normalizing    
    norm_signal = Preprocessing(fs, x)
    
    # estimation Power Spectral Density 
    fq, Pxx = signal.welch(norm_signal, fs, nperseg=2048)
    Pxx_dB = 10*np.log10(Pxx)

    # sample position estimation
    n1 = (np.abs(fq - f1)).argmin()
    n2 = (np.abs(fq - f2)).argmin()
    
    # extract interval from 0-2000 Hz
    seg1 = Pxx_dB[0:n1]
    
    # extract interval from 2000-5000 Hz
    seg2 = Pxx_dB[n1:n2]
              
    # computing max SPL (Sound Preasure Level)
    SPL02 = max(seg1)
    SPL25 = max(seg2)
    HammIndex = SPL02-SPL25
    
    return HammIndex,fq,Pxx_dB
    

def Preprocessing(fs, x):
    
    # ensure only one channel
    if x.ndim>1:
        mono_signal = 1/2*(x[:,0] + x[:,1])
    else:
        mono_signal = x
        
    # normalizing [-1, 1]
    norm_signal = mono_signal/max(mono_signal) 
    
    # check minimum signal duration
    T = 5 # seconds        
    t = len(norm_signal)/fs # Signal duration in seconds
    
    # check signal duration
    if t<T:
        sys.exit('Signal duration must be greater than 5s')                                      
    
    # check sampling frequency
    if fs <12000:
        sys.exit('Sampling Rate fs must be greater than 12KHz')                           
    
    return norm_signal