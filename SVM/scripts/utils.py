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
    theta_band = (4,7)
    alpha_band = (7,13)
    beta_band = (13,30)
    theta = butter_bandpass_filter(fft_data, theta_band[0], theta_band[1], fs)
    alpha = butter_bandpass_filter(fft_data, alpha_band[0], alpha_band[1], fs)
    beta = butter_bandpass_filter(fft_data, beta_band[0], beta_band[1], fs)
    return theta, alpha, beta
