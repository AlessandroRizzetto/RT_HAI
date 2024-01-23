import librosa
import numpy as np
from scipy.fft import fft
from scipy.stats import linregress

def transform(info, sin, sout, sxtras, board, opts, vars):   
    npin = np.asmatrix(sin)
    
    sout[0] = calculate_slope(npin, sin.sr)

def calculate_slope(audio_samples, sample_rate, regression_method=True):
    # Calcola la trasformata di Fourier dei campioni audio con input una np.matrix
    spectrum = np.abs(fft(audio_samples))
    
    # Calcola le frequenze corrispondenti alla trasformata di Fourier
    frequencies = librosa.core.fft_frequencies(sr=sample_rate, n_fft=spectrum.shape[0]*2-1) #np.fft.fftfreq(len(audio_samples), 1 / sample_rate)

    if regression_method:
        # Seleziona le bande di frequenza tra 50Hz e 100Hz
        selected_band = (frequencies >= 50) & (frequencies <= 100)

        # Calcola l'energia media nelle bande di frequenza selezionate
        energy = np.mean(np.square(spectrum[selected_band, :]), axis=1)

        # Calcola lo slope utilizzando la regressione lineare
        slope, _, _, _, _ = linregress(frequencies[selected_band], energy)
    else:
        # Trova gli indici delle frequenze desiderate
        index_50hz = np.argmax(frequencies >= 50)
        index_100hz = np.argmax(frequencies >= 100)

        # Calcola la pendenza tra 50Hz e 100Hz
        slope = (spectrum[index_100hz] - spectrum[index_50hz]) / (frequencies[index_100hz] - frequencies[index_50hz])

    return slope