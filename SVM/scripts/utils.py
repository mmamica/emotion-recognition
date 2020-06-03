import numpy as np
from scipy import signal, fft

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def extract_bands(data, fs):
    fft_data = fft.fft(data)
    delta_band = (0.5,4)
    theta_band = (4,7)
    alpha_band = (7,13)
    beta_band = (13,30)
    gamma_band = (30,60)
    delta = butter_bandpass_filter(fft_data, delta_band[0], delta_band[1], fs)
    theta = butter_bandpass_filter(fft_data, theta_band[0], theta_band[1], fs)
    alpha = butter_bandpass_filter(fft_data, alpha_band[0], alpha_band[1], fs)
    beta = butter_bandpass_filter(fft_data, beta_band[0], beta_band[1], fs)
    gamma = butter_bandpass_filter(fft_data, gamma_band[0], gamma_band[1], fs)
    return delta, theta, alpha, beta, gamma


def total_wavelet_energy(bands):
    energy = 0
    for b in bands:
        energy += np.sum(b**2)
    return energy


def wavelet_entropy(signal, energy):
    p = np.sum(signal**2)/energy
    return -p*np.log(p)


def add_data_channels(data, sample, output):
    return np.hstack((output, data[sample][0].reshape(8064,1), data[sample][2:6].reshape(8064,4), data[sample][16].reshape(8064,1), data[sample][18:23].reshape(8064,5)))
