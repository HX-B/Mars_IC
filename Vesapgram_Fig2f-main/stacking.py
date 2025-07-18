# stacking.py
# module: vespy.stacking
# Functions for applying various stacking methods to seismic data

from utils import get_event_coordinates_list
import numpy as np
import scipy.signal as sig
import cmath

def degrees_to_radians(theta):
    return theta * np.pi / 180

def resolve_slowness_vector(s, baz):
    '''
    Resolves a scalar slowness and backazimuth into the x and y components of the two-dimensional slowness vector.

    Parameters
    ----------
    s  : float
        Magnitude of slowness vector, in s / km
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)

    Returns
    -------
    (s_x, s_y) : tuple
        Tuple containing the magnitude of the x and y components of the 2d slowness vector, in s / km.
    '''

    baz_rad = np.deg2rad(baz)

    s_x = s * np.sin(baz_rad)
    s_y = s * np.cos(baz_rad)

    return s_x, s_y

def get_shifts(st,distances, s, baz,):
    '''
    Calculates the shifts (as an integer number of samples in the time series) for every station in a stream of time series seismograms for a slowness vector of given magnitude and backazimuth.

    The shift is that which needs to be applied in order to align an arrival (arriving with slowness s and backazimuth baz) with the same arrival at the array reference point (the location of the station that makes up the first trace in the stream).

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    s  : float
        Magnitude of slowness vector, in s / km
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)

    Returns
    -------
    shifts : list
        List of integer delays at each station in the array, also length K
    '''
    theta = [] # Angular position of each station, measured clockwise from North
    r = [] # Distance of each station

    # First station is reference point, so has zero position vector
    # theta.append(0.0)
    # r.append(0.0)

    geometry = get_event_coordinates_list(st,distances) # in deg

    # For each station, get distance from array reference point (first station), and the angular displacement clockwise from north
    for station in geometry[0:]:
        r_x = station[0] # x-component of position vector
        r_y = station[1] # y-component of position vector

        # theta is angle c/w from North to position vector of station; 
        # need to compute diffently for each quadrant
        if r_x == 0 and r_y == 0:
            theta.append(0.0)
        elif r_x > 0 and r_y == 0:
            theta.append(90.0)
        elif r_x < 0 and r_y == 0:
            theta.append(270.0)
        elif r_x >= 0 and r_y > 0:
            theta.append(np.degrees(np.arctan(r_x/r_y)))
        elif r_x >= 0 and r_y < 0:
            theta.append(180.0 + np.degrees(np.arctan(r_x/r_y)))
        elif r_x < 0 and r_y < 0:
            theta.append(180.0 + np.degrees(np.arctan(r_x/r_y)))
        else:
            theta.append(360.0 + np.degrees(np.arctan(r_x/r_y)))

        r.append(np.sqrt(r_x**2 + r_y**2))

    # Find angle between station position vector and slowness vector in order to compute dot product

    # Angle between slowness and position vectors, measured clockwise
    phi = [180 - baz + th for th in theta]
    sampling_rate = st[0].stats.sampling_rate

    shifts = []

    # Shift is dot product. The minus sign is because a positive time delay needs to be corrected by a negative shift in order to stack
    for i in range(0, len(st)):

        shifts.append(-1 * int(round(r[i] * s * np.cos(np.radians(phi[i])) * sampling_rate)))
    return shifts

def linear_stack(st,distances, s, baz):
    '''
    Returns the linear (delay-and-sum) stack for a seismic array, for a beam of given slowness and backazimuth.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    s  : float
        Magnitude of slowness vector, in s / km
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)

    Returns
    -------
    stack : NumPy array
        The delay-and-sum beam at the given slowness and backazimuth, as a time series.
    '''

    # Check that each channel has the same number of samples, otherwise we can't construct the beam properly
    assert len(set([len(tr) for tr in st])) == 1, "Traces in stream have different lengths, cannot stack."

    nsta = len(st)

    shifts = get_shifts(st,distances, s, baz)

    shifted_st = st.copy()
    for i, tr in enumerate(shifted_st):
        tr.data = np.roll(tr.data, shifts[i])

    stack = np.sum([tr.data for tr in shifted_st], axis=0) / nsta

    return stack

def nth_root_stack(st,distances, s, baz,n):
    '''
    Returns the nth root stack for a seismic array, for a beam of given slowness and backazimuth.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    s  : float
        Magnitude of slowness vector, in s / km
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)
    n : int
        Order of the nth root process (n=1 just yields the linear vespa)

    Returns
    -------
    stack : NumPy array
        The nth root beam at the given slowness and backazimuth, as a time series.
    '''
    # Check that each channel has the same number of samples, otherwise we can't construct the beam properly
    assert len(set([len(tr) for tr in st])) == 1, "Traces in stream have different lengths, cannot stack."

    nsta = len(st)

    shifts = get_shifts(st,distances, s, baz)

    stack = np.zeros(st[0].data.shape)
    for i, tr in enumerate(st):
        stack += np.roll(pow(abs(tr.data), 1./n) * np.sign(tr.data), shifts[i]) # Shift data in each trace by its offset

    stack /= nsta
    stack = pow(abs(stack), n) * np.sign(stack)

    return stack

def phase_weighted_stack(st,distances, s, baz, n=1):
    '''
    Calculates the phase-weighted stack for seismograms in the stream. n is the order of the phase-weighting.

    n should be an integer >= 0. n = 0 corresponds with no phase weighting, i.e. just the linear stack.

    Parameters
    ----------
    st: ObsPy Stream object
        The stream of seismograms for the array for a particular event
    s : float
        Magnitude of slowness vector, in s / km
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)
    n : number
        Order of the phase-weighted stacking to be applied, default 1. 
        Should be int, n >= 0.

    Returns
    -------
    stack : NumPy array
        Phase-weighted stack for the given event at the array

    Notes
    -----
    The phase-weighted stack weights the data from each seismogram by its instantaneous phase. This phase information is obtained from Hilbert transform of the data:-

    .. math:: S(t) = d(t) + iH[d(t)] = A(t)e^{i\Phi(t)}

    where :math:`S(t)` is the analytic time series of the data :math:`d(t)`, and :math:`A(t)` and :math:`\Phi(t)` respectively represent the instantaneous amplitude and phase of the data.

    The phase stack is then defined as:-

    .. math:: c(t) = \\frac{1}{N}|\sum_{k=1}^Ne^{i\Phi_k(t)}|^n

    where :math:`N` is the total number of seismograms in the array, and :math:`n` is the order of the weighting. This is then applied to the linear stack to obtain the phase-weighted stack, :math:`\hat{d}(t)`:-

    .. math:: \hat{d}'(t) = \\frac{1}{N}\sum_{j=1}^Nd_j(t) |\\frac{1}{N}\sum_{k=1}^Ne^{i\Phi_k(t)}|^n

    where :math:`d_j(t)` is the time series data from the :math:`j` th seismogram in the array.

    '''
    assert n >= 0, "n should be an integer >= 0."

    N = len(st) # Number of seismograms in stream
    sampling_rate = st[0].stats.sampling_rate

    shifts = get_shifts(st,distances, s, baz)

    # Calculate Hilbert transforms for analytic signal to use in phase calculations
    hilbert_transforms = []

    for tr in st:
        tr.data = np.require(tr.data, dtype=np.float32) # Due to bug in ObsPy, trace data somehow converted to big-endian floats which the fFT can't handle. Need to enforce little-endianness (http://lists.swapbytes.de/archives/obspy-users/2014-December.txt)
        hilbert_transforms.append(sig.hilbert(tr.data))

    # Now calculates the instantaneous phases for the sream
    phases = []

    for hilbert_transform in hilbert_transforms:
        phases.append(np.array([cmath.phase(z) for z in hilbert_transform]))

    # Need to apply correct delay times to each channel before calculating phase stack
    for i, channel in enumerate(phases):
        channel = np.roll(channel, shifts[i])

    phases = np.array(phases)

    # Phase stack
    phase_weightings = abs(np.sum(np.exp(phases*1j), axis=0)) / N
    phase_weightings = np.array(phase_weightings)

    # Start with linear stack, then apply phase-weighting
    linstack = linear_stack(st,distances, s, baz)

    stack = linstack * phase_weightings**n

    return stack
