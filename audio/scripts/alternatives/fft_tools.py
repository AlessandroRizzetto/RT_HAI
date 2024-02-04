import numpy as np

from scipy.fft import fft

def fft_magnitude(window):
   return abs(fft(window))