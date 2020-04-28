import numpy as np
from scipy.signal import butter, lfilter, welch
import math as m

def cart2sph(x, y, z):
    x2_y2 = x**2 + y**2
    r = m.sqrt(x2_y2 + z**2)                    # r
    elev = m.atan2(z, m.sqrt(x2_y2))            # Elevation
    az = m.atan2(y, x)                          # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    return rho * m.cos(theta), rho * m.sin(theta)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def frequency_bands(data):
    fft = np.fft.fft(data)
    cfft = fft*np.conjugate(fft)
    alpha = butter_bandpass_filter(cfft[:], 8.1, 13.0, 128)
    beta = butter_bandpass_filter(cfft[:], 13.1, 30.0, 128)
    gamma = butter_bandpass_filter(cfft[:], 30.1, 40, 128)
    theta = butter_bandpass_filter(cfft[:], 4.1, 8.0, 128)
    return alpha, beta, gamma, theta

def fft_to_psd(data,fs):
    l = len(data)
    return (1/(l*fs))*np.abs(data)*np.abs(data)

def average(data):
    l = len(data)
    return np.sum(data)/l