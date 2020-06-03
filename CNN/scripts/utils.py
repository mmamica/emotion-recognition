import numpy as np
from scipy.signal import welch, periodogram


def count_azimuth(center_point, point):
    '''
    calculate azimuth of given points
    :param center_point: relative, center point
    :param point: point for which azimuth is calculated
    :return: azimuth value
    '''
    long_center, lat_center = center_point[0], center_point[1]
    long, lat = point[0], point[1]
    if(long_center == long and lat_center == lat):
        return 0
    tan_azimuth = (np.cos(lat)*np.sin(long - long_center))/(np.cos(lat_center)*np.sin(lat) - np.sin(lat_center)*np.cos(lat)*np.cos(long-long_center))
    return np.arctan(tan_azimuth)


def count_distance(center_point, point):
    '''
    calculate distance between points
    :param center_point: relative, center point
    :param point: point for which distance is calculated
    :return:
    '''
    long_center, lat_center = center_point[0], center_point[1]
    long, lat = point[0], point[1]
    cos_distance = np.sin(lat_center)*np.sin(lat) + np.cos(lat_center)*np.cos(lat)*np.cos(long-long_center)
    return np.arccos(cos_distance)


def psd_bands_welch(data, sf):
    '''
    extract alpha, beta, theta, gamma bands and calculate average power spectral density for each band with Welch method
    :param data: EEG data
    :param sf: frequency sampling
    :return: average PSDs for all bands
    '''
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
    '''
    extract alpha, beta, theta, gamma bands and calculate average power spectral density for each band with periodogram
    :param data: EEG data
    :param sf: frequency sampling
    :return: average PSDs for all bands
    '''
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