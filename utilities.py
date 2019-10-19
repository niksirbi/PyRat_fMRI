# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:00:51 2019

@author: nsirmpilatze
"""

import os
import numpy as np

# small value to be added to certain denominators to avoid division by 0
EPSILON = 10**(-8)


def gaus(x, A, mu, sigma):
    """Gaussian function

    Parameters
    ----------
    x : 1d-array
        Values ar which function is evaluated
    A : float
        Amplitude
    mu : float
        Mean
    sigma: float
        Standard deviation

    Return
    ------
    G: 1d-array
        Gaussian function of length len(x)
    """
    G = A*np.exp(-np.power(x - mu, 2.) / (2*np.power(sigma, 2.)))
    return G


def read_nifti(file):
    """Loads a NIFTI file

    Parameters
    ----------
    file: string
        path to NIFTI or NIFTI_GZ file

    Return
    ------
    result: dictionary with the following keys
        data: numpy matrix containing image data
        affine: NIFTI affine (from header)
        header: NIFTI header
    """

    import nibabel as nib

    handle = nib.load(file)
    data = handle.get_data()
    affine = handle.affine
    header = handle.header

    result = {'data': data,
              'affine': affine,
              'header': header}

    return result


def mat_pearsonr(A, B):
    """Calculate row-wise Pearson's correlation between 2 2d-arrays

    Parameters
    ----------
    A: 2d-array
        shape N x T
    B: 2d-array
        shape M x T

    Return
    ------
    R: 2d-array
        N x M shaped correlation matrix between all row combinations of A and B
    """

    #  Subtract row-wise mean from input arrays
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get correlation coefficient
    numerator = np.dot(A_mA, B_mB.T)
    denominator = np.sqrt(np.dot(ssA[:, None], ssB[None])) + EPSILON
    R = numerator / denominator

    return R


def butter_filter(x, cutoff, fs, which='low', order=5):
    """ digital Butterworth filter

    Parameters
    ----------
    x: 1d-array
        input signal
    cutoff: float
        cutoff frequency
    fs: float
        sampling frequency
    which: str
        "low"/"high" for lowpass/highpass respectively
    order: int,
        order of filter

    Return
    ------
    y: 1d-array
    filtered signal
    """

    from scipy.signal import butter, filtfilt

    nyquist = 0.5*fs
    normal_cutoff = cutoff/nyquist
    b, a = butter(order, normal_cutoff, btype=which, analog=False)
    y = filtfilt(b, a, x)
    # compensate for offset at origin
    diff = x[0] - y[0]
    y = y + diff

    return y


def bandpower(data, fs, band, method='welch', window_sec=None, relative=False):
    """Computes the average power of the signal x in a specific frequency band

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    fs : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    from mne.time_frequency import psd_array_multitaper

    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * fs
        else:
            nperseg = (2 / low) * fs

        freqs, psd = welch(data, fs, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, fs, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp
