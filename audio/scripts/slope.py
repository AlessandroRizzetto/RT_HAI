import numpy as np
from scipy.fft import fft

def transform(info, sin, sout, sxtras, board, opts, vars):   
    npin = np.asmatrix(sin)
    
    sout[0] = calculate_slope(npin, sin.sr)

def calculate_slope(audio_samples, sample_rate):
    # Calcola la trasformata di Fourier dei campioni audio con input una np.matrix
    spectrum = np.abs(fft(audio_samples))

    # Calcola le frequenze corrispondenti alla trasformata di Fourier
    frequencies = np.fft.fftfreq(len(audio_samples), 1 / sample_rate)

    # Trova gli indici delle frequenze desiderate
    index_50hz = np.argmax(frequencies >= 50)
    index_100hz = np.argmax(frequencies >= 100)

    # Calcola la pendenza tra 50Hz e 100Hz
    slope = (spectrum[index_100hz] - spectrum[index_50hz]) / (frequencies[index_100hz] - frequencies[index_50hz])

    return slope