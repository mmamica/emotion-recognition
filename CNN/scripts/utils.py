import numpy as np
from scipy.signal import welch, periodogram
import math as m

def cart2sph(x, y, z):
    x2_y2 = x**2 + y**2
    r = m.sqrt(x2_y2 + z**2)                    # r
    elev = m.atan2(z, m.sqrt(x2_y2))            # Elevation
    az = m.atan2(y, x)                          # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    return rho * m.cos(theta), rho * m.sin(theta)

def psd_bands_welch(data, sf):
    win = 0.5 * sf
    freqs, psd = welch(data, sf, nperseg=win)

    # Define lower and upper limits
    low_theta, high_theta = 4.0, 8.0
    low_alpha, high_alpha = 8.0, 13.0
    low_beta, high_beta = 13.0, 30.0
    low_gamma, high_gamma = 30.0, 40.0

    # Find intersecting values in frequency vector
    idx_theta = np.logical_and(freqs > low_theta, freqs <= high_theta)
    idx_alpha = np.logical_and(freqs > low_alpha, freqs <= high_alpha)
    idx_beta = np.logical_and(freqs > low_beta, freqs <= high_beta)
    idx_gamma = np.logical_and(freqs > low_gamma, freqs <= high_gamma)

    # Take averaged psds
    psd_theta = np.average(psd[idx_theta])
    psd_alpha = np.average(psd[idx_alpha])
    psd_beta = np.average(psd[idx_beta])
    psd_gamma = np.average(psd[idx_gamma])

    return psd_theta, psd_alpha, psd_beta, psd_gamma

def psd_bands_periodogram(data, sf):
    freqs, psd = periodogram(data, sf)

    # Define lower and upper limits
    low_theta, high_theta = 4.0, 8.0
    low_alpha, high_alpha = 8.0, 13.0
    low_beta, high_beta = 13.0, 30.0
    low_gamma, high_gamma = 30.0, 40.0

    # Find intersecting values in frequency vector
    idx_theta = np.logical_and(freqs > low_theta, freqs <= high_theta)
    idx_alpha = np.logical_and(freqs > low_alpha, freqs <= high_alpha)
    idx_beta = np.logical_and(freqs > low_beta, freqs <= high_beta)
    idx_gamma = np.logical_and(freqs > low_gamma, freqs <= high_gamma)

    # Take averaged psds
    psd_theta = np.average(psd[idx_theta])
    psd_alpha = np.average(psd[idx_alpha])
    psd_beta = np.average(psd[idx_beta])
    psd_gamma = np.average(psd[idx_gamma])

    return psd_theta, psd_alpha, psd_beta, psd_gamma