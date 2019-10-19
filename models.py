
# coding: utf-8
"""
@author: nsirmpilatze

This module provided utility functions for constructing
fMRI regressors:
1. two_gamma_hrf: build HRF
2. plot_hrf: display HRF
3. block: define block design
4. events: define event-related design
5. hrf_convolve: convolve design with HRF
"""

# import modules

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as sps
import seaborn as sns
sns.set(style='ticks', context='notebook', font_scale=1.5)
cp = sns.color_palette()


def two_gamma_hrf(length=32,
                  TR=2,
                  peak_delay=6,
                  under_delay=16,
                  peak_disp=1,
                  under_disp=1,
                  p_u_ratio=6,
                  normalize=True,
                  ):
    """ HRF function from sum of two gamma PDFs

    It is a *peak* gamma PDF (with location `peak_delay` and
    dispersion `peak_disp`), minus an *undershoot* gamma PDF
    (with location `under_delay` and dispersion `under_disp`,
    and divided by the `p_u_ratio`).

    Parameters
    ----------
    length: float
        length of HRF in seconds
    TR : float
        repetition (sampling) time in seconds
    peak_delay : float, optional
        delay of peak
    peak_disp : float, optional
        width (dispersion) of peak
    under_delay : float, optional
        delay of undershoot
    under_disp : float, optional
        width (dispersion) of undershoot
    p_u_ratio : float, optional
        peak to undershoot ratio.  Undershoot divided by this value before
        subtracting from peak.
    normalize : {True, False}, optional
        If True, divide HRF values by their sum before returning.

    Returns
    -------
    hrf : array
        vector of samples from HRF at TR intervals
    """

    t = np.linspace(0, length, length/TR)
    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
            if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    # gamma.pdf only defined for t > 0
    hrf = np.zeros(t.shape, dtype=np.float)
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t,
                         peak_delay / peak_disp,
                         loc=0,
                         scale=peak_disp)
    undershoot = sps.gamma.pdf(pos_t,
                               under_delay / under_disp,
                               loc=0,
                               scale=under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio
    if not normalize:
        return hrf
    return hrf / np.max(hrf)


def plot_hrf(hrf, TR=2):

    """ Plots HRF

    Parameters
    ----------
    hrf : array-like
        output of "two_gamma_hrf" function
    TR : float
        TR in seconds (sampling interval for HRF)

    Returns
    -------
    Displays a figure with the HRF
    """

    n = len(hrf)
    time = np.linspace(0, n*TR, n/TR)
    plt.figure(figsize=(6, 4))
    plt.plot(time, hrf, 'o-', color=cp[3], lw=2)
    plt.xlabel('Time (s)')
    plt.ylim(-0.3, 1.2)
    plt.xlim(0, n*TR)
    plt.yticks([0, 1])
    plt.xticks(range(0, n*TR, 5), fontsize=18)
    plt.grid(axis='both')
    plt.title('HRF')
    sns.despine()
    plt.show()


def block(base=18,
          on=12,
          off=18,
          cycles=8,
          TR=2):

    """ Creates boxcar regressor for block design
    Parameters are given in seconds
    'Base', 'on', 'off' must be integer multiples of TR

    Parameters
    ----------
    base : float
        baseline duration
    on : float
        stimulus block duration
    off: float
        rest block duration
    cycles: int
        number of on-off cycles
    TR : float
        repetition (sampling) time in seconds

    Returns
    -------
    A boxcar model of the block design in volumes
    """

    # Convert seconds to volumes (TRs)
    for v in [base, on, off]:
        if v % TR != 0:
            raise ValueError("base, on, off must be multiples of TR")
    base, on, off = int(base//TR), int(on//TR), int(off//TR) 

    # Build model
    baseline = np.zeros(base)
    repeat = np.tile(np.append(np.ones(on), np.zeros(off)), cycles)
    y = np.hstack((baseline, repeat))

    return y


def events(length, starts, dur):

    """ Creates boxcar regressor for event-related design
    Parameters are given in seconds
    'length', 'dur' must be integer multiples of TR

    Parameters
    ----------
    length: float
        duration of fMRI run
    starts: a list of floats
        onset times for events
    dur: float or list of floats
        duration of each event

    Returns
    -------
    A boxcar model of the event-related design in volumes
    """

    y = np.zeros(length)
    for s in starts:
        y[s:s+dur] = np.ones(dur)

    return y


def hrf_convolve(model, hrf):

    """ Convolves HRF with boxcar regressor
    Parameters are in volumes

    Parameters
    ----------
    model: array-like
        boxcar model
    hrf: array-like
        output of "two_gamma_hrf" function

    Returns
    -------
    A HRF-convolved model in volumes
    """

    y = np.convolve(model, hrf, mode='full')
    y = y/np.max(y)
    volumes = len(model)

    return y[:volumes]
