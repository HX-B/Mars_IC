import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

from stacking import linear_stack, nth_root_stack, phase_weighted_stack
from stats import n_power_vespa, f_vespa, pw_power_vespa

def vespagram(st,distances, smin, smax, ssteps, baz=270, winlen=20, stat='power', stack="pws", n=1):
    '''
    Calculates the vespagram for a seismic array over a given slowness range, for a single backazimuth, using the statistic specified.

    The chosen statistic is calculated as a function of time (in s) and slowness (in s/km). This may be:-

    * 'amplitude' - the raw amplitude of the linear or nth root stack at each time and slowness step;
    * 'power' - the power in the linear or nth root beam calculated over a time window (length winlen) around each time step for each slowness;
    * 'F' - the F-statistic of the beam calculated over a time window (length winlen) around each time step for each slowness.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    smin  : float
        Minimum magnitude of slowness vector, in s / km
    smax  : float
        Maximum magnitude of slowness vector, in s / km
    ssteps  : int
        Integer number of steps between smin and smax for which to calculate the vespagram
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)
    winlen : int
        Length of Hann window over which to calculate the power.
    stat : string
        Statistic to use for plotting the vespagram, either 'amplitude', 'power', or 'F'
    phase_weighting : Boolean
        Whether or not to apply phase-weighting to the stacks in the vespagram.
    n : int
        Order for the stack, or if phase_weighting==True, order of the weighting applied.


    Returns
    -------
    vespagram_data : NumPy array
        Array of values for the chosen statistic at each slowness and time step. Dimensions: ssteps*len(tr) for traces tr in st.
    '''

    assert stat == 'amplitude' or stat == 'power' or stat == 'F', "'stat' argument must be one of 'amplitude', 'power' or 'F'"

    vespagram_data = np.array([])

    try:
        if stat == 'amplitude':
            if stack=="pws":
                vespagram_data = np.array([phase_weighted_stack(st,distances, s, baz, n) for s in np.linspace(smin, smax, ssteps)])
            elif stack=="nroot": 
                vespagram_data = np.array([nth_root_stack(st,distances, s, baz, n) for s in np.linspace(smin, smax, ssteps)])

        elif stat == 'power':
            if stack=="pws":
                vespagram_data = np.array([pw_power_vespa(st,distances, s, baz, n, winlen) for s in np.linspace(smin, smax, ssteps)])
            elif stack=="nroot":
                vespagram_data = np.array([n_power_vespa(st,distances, s, baz, n, winlen) for s in np.linspace(smin, smax, ssteps)])

        elif stat == 'F':
            vespagram_data = np.array([f_vespa(st,distances, s, baz, winlen, n) for s in np.linspace(smin, smax, ssteps)])

    except AssertionError as err:
        raise err

    return vespagram_data