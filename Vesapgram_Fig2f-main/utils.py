# utils.py
# module: vespy.utils
# General purpose utilities for seismic array analysis

import numpy as np
import mpl_toolkits
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from obspy.core import Trace
from obspy.taup import TauPyModel
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException

G_KM_DEG = 59.15 # km / deg, Conversion factor for converting angular great circle distance (in degrees) into km on the surface

class Phase:
    '''
    Class for handling phase arrivals.

    Similar to obspy.core.event.Arrival, which does the same thing but with a few more parameters I don't need

    :type name: string
    :param name: Name of seismic phase
    :type time: float
    :param time: Time after event origin of arrival in seconds
    :type slowness: float
    :param slowness: Slowness of seismic phase in seconds / kilometre

    '''

    def __init__(self, phase_name, arrival_time, slowness_s_km):
        self.name = phase_name
        self.time = arrival_time
        self.slowness = slowness_s_km

    def __str__(self):
        return_str = "%s, %.1f, %.3f" % (self.name, self.time, self.slowness)
        return return_str

def traceify(data_array, ref_trace, name="NEW_TRACE", channel=None):
    '''
    Turns an array into an ObsPy Trace object, using the reference trace.

    The new trace inherits the time data all stats (except 'station' and 'channel', which are left for the user to define) from the reference trace.

    Parameters
    ----------
    data_array: NumPy array-like object
        The data to be turned into an ObsPy Trace. Should be 1D time series numerical data of the same length as the reference trace, e.g. F-statistic, power, envelope etc.
    ref_trace: ObsPy Trace object
        Reference time series data to use when creating the new trace
    name: string
        String to identify the new trace, e.g. 'F-STAT', 'POWER', 'ENVELOPE'. Occupies the 'station' SAC header variable.
    channel: string
        Data channel to identify the component of the time series data the new trace represents (if applicable)

    Returns
    -------
    new_trace: ObsPy Trace object
        An ObsPy Trace containg  the data in data_array, and the time data and stats from ref_trace

    '''

    new_trace = Trace(data_array)
    new_trace.stats = ref_trace.stats.copy()
    new_trace.stats.station = name

    if channel is not None:
        new_trace.stats.channel = channel

    return new_trace

#### change by bihx
def get_event_coordinates_list(stream,distances,radiu=3389.5):
    '''
    Calculates the x, y, z coordinates of stations in a seismic array relative to a reference point for a given stream of SAC seismographic data files.

    The reference point will be taken as the coordinates of the first station in the stream.

    Parameters
    ----------
    stream : ObsPy Stream object
        The stream of seismograms for the array

    Returns
    -------
    xyz_rel_coords : NumPy Array
        x, y, z coordinates for each station in the array, measured in metres relative to the first station in the stream
    '''
    # Get coordinates of stations in array. Looks for SAC headers stla, stlo, stel.
    coords = []
    # Get station name, gcarc, baz

    for i,trace in enumerate(stream):
        # coords.append((Dislist[i], trace.stats.sac.baz))
        coords.append((distances[i], 90))

    coords = np.array(coords)

    xyz_coords = []
    for gcarc,baz in coords:
        #dist = gcarc * (2*np.pi*radiu / 360)
        dist = gcarc 
        x = dist* np.sin(np.deg2rad(baz))
        y = dist* np.cos(np.deg2rad(baz))
        xyz_coords.append((x, y))

    # Get coordinates relative to the first event
    # x_0,y_0 = xyz_coords[0]

    xyz_coords = np.array(xyz_coords)
    # x_0,y_0 = [np.mean(xyz_coords[:,0]),90]
    x_0,y_0 = [29.0,90]   #### 固定某一震中距29.0 保持一致

    xyz_rel_coords = []
    for x, y in xyz_coords:
        x -= x_0
        y -= y_0
        xyz_rel_coords.append((x, y))
    xyz_rel_coords = np.array(xyz_rel_coords)
    return xyz_rel_coords



