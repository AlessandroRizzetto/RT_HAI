import librosa
import numpy as np
import fft_tools as ft

def transform(info, sin, sout, sxtras, board, opts, vars):   
    npin = np.asmatrix(sin)
    
    sout[0] = alpha_ratio(ft.fft_magnitude(npin), sin.sr)

def alpha_ratio(fft_magnitude, sr):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        fft_magnitude:            the abs(fft) of the current frame
    """
    freqs = librosa.core.fft_frequencies(sr=sr, n_fft=fft_magnitude.shape[0]*2-1)

    # Definisci le bande di frequenza
    low_band = (freqs >= 0) & (freqs < 1000)
    high_band = (freqs >= 1000) & (freqs < 5000)

    # Calcola l'energia nelle bande di frequenza
    low_band_energy = np.mean(np.square(fft_magnitude[low_band, :]))
    high_band_energy = np.mean(np.square(fft_magnitude[high_band, :]))

    # Calcola l'Alpha Ratio
    alpha_ratio = low_band_energy / high_band_energy if high_band_energy != 0 else 0

    return alpha_ratio