# plot vespagram, take Fig.2f as an example

import os
import gc
import re
import obspy
from obspy.signal.filter import envelope
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import vespagram as VesPa

def z_score(x, axis):
    """
    Normalize individual trace, as Eq.1 in the Methods introduced.
    """
    x  = np.array(x).astype(float)
    xr = np.rollaxis(x, axis=axis)
    xr -= np.mean(x, axis=axis)
    xr /= np.std(x, axis=axis)
    return x

def envelope_smooth(envelope_window_in_sec, tr, mode='valid'):
    """
    Obtain the envelope of individual trace
    """
    tr_env      = tr.copy()
    tr_env.data = envelope(tr_env.data)
    w           = np.ones(int(envelope_window_in_sec / tr.stats.delta))
    w          /= w.sum()
    tr_env.data = np.convolve(tr_env.data, w, mode=mode)
    return tr_env

def create_ax_axbar(fig,cord=[0.15,0.10,0.80,0.85],
                    labelsize=20,fontsize=20,lgsize=20):
    """
    create the figure layout
    """
    rect = fig.patch
    rect.set_facecolor('white')
    parameters = {'axes.labelsize':labelsize,'axes.titlesize': 12,
                  'xtick.labelsize':labelsize,'ytick.labelsize':labelsize}
    plt.rcParams.update(parameters)
    plt.rcParams['font.size'] = fontsize
    plt.rc('legend', fontsize=lgsize)
    ax = fig.add_axes(cord)
    ax.minorticks_on()
    ax.tick_params(which='major',length=6)
    ax.tick_params(which='minor',length=3)
    ax.tick_params(top=False,bottom=True,left=True,right=False)

    ax_bar = fig.add_axes([cord[0]+cord[-2]+0.02,
                           cord[1],
                           0.02,
                           cord[-1]])
    ax_bar.axes.yaxis.set_ticklabels([])
    return ax,ax_bar

##===================================================================
##===================================================================

## =========== 1. obtain the synthetics generate by AxiSEM ======== 
DataPath     = "./DataSyn/SAC/"
Receiverpath = os.path.join("./DataSyn/","receiver_gll_locs_explosion.dat")
receiver     = pd.read_csv(Receiverpath,names=["ID","dist"],usecols=[0,1],sep="\s+",header=0)
Dist_syn     = receiver["dist"].to_numpy()
Index        = np.arange(54,90)     #### epicentral distance range: 27-40 degrees

Phase           = ["P","PKiKP",]    #### target phase: PKiKP, align on direct P-wave
colors          = ["orange","red"]

channel         = "Z"               #### vertical component
freqmin,freqmax = 0.1,0.5           #### frequency band for bandpass filter, which is limited by the mesh resolution, here, we run up to 2s.
bsec,esec       = 400,800           #### cut time window to do vespagram analysis

## ------------ read data and save into a obspy.Stream --------
Trace           = []
Dist_obs        = np.array([])
for i in Index:
    dist    = Dist_syn[i]
    fname   = receiver["ID"].to_numpy()[i]
    file    = os.path.join(DataPath,"{}.BH{}.SAC".format(fname,channel))
    otrace  = obspy.read(file)[0]
    otrace.filter("bandpass",freqmin=freqmin, freqmax=freqmax,corners=4, zerophase=True)
    tstart  = otrace.stats.starttime
    sampling_rate = otrace.stats.sampling_rate
    index         = int((esec-bsec)*sampling_rate)
    try:
        t1         = otrace.stats.sac.t1     ## P arrival
        trace      = otrace.slice(tstart+t1+bsec,tstart+t1+esec)
        trace.data = trace.data[:index+1]
        # if envolope ######
        trace.data = z_score(trace.data,axis=0)
        tr_env     = envelope_smooth(envelope_window_in_sec=5., tr=trace,mode='same')
        trace      = tr_env
        Trace.append(trace)
        Dist_obs = np.append(Dist_obs,
                            [trace.stats.sac.gcarc])  
    except Exception:
        print("{} has wrong!".format(dist))
        continue
Stream = obspy.Stream(Trace)

## =================== 2. Calculate vespagram  =========================== 
## :parameters, see details in vespagram.py
smin   = -10 
smax   = 0
ssteps = 50 
stack  = "nroot"
stat   = "power"
n      = 4

st             = Stream.copy()
sampling_rate  = st[0].stats.sampling_rate
vespagram_Data = VesPa.vespagram(st=Stream,distances=Dist_obs,
                                    smin=smin,smax=smax,ssteps=ssteps,
                                    stat=stat,stack=stack,n=n,)

plot_bsec,plot_esec =  550,700  ## cut a short time window to show the vespagram 
plot_bdex,plot_edex = [int((plot_bsec-bsec) * sampling_rate),int((plot_esec-bsec) * sampling_rate)]
vespagram_data = vespagram_Data[:,plot_bdex:plot_edex]
vespagram_data = vespagram_data / np.max(vespagram_data)   ## normalize


## =================== 3. Plot vespagram  =========================== 
fig         = plt.figure(figsize=(8,4),constrained_layout=True)
ax,ax_bar   = create_ax_axbar(fig,cord=[0.15,0.10,0.78,0.85],labelsize=18,fontsize=18,lgsize=18)

tmin  = 0
tmax  = st[0].times()[-1]
xtime = st[0].times()[plot_bdex:plot_edex]+bsec
vmin  = np.min(vespagram_data)
vmax  = np.max(vespagram_data) 
levels = np.linspace(vmin,vmax,50)
norm   = mpl.colors.Normalize(vmin=0.5, vmax=1.0)

cmap  = "viridis"
cax   = ax.contourf(xtime, np.linspace(smin, smax, ssteps), vespagram_data[:, :],levels,
                    cmap=cmap,norm=norm)
lab   = "{} ".format(stat.capitalize()) 
cbar  = mpl.colorbar.ColorbarBase(ax_bar,cmap=cmap,
                                  norm=norm,label=lab,
                                  orientation="vertical")
ax.set_xlim(plot_bsec,plot_esec)
ax.set_ylim(smin,smax)

ax.set_xlabel("Time relative to P (s)")
ax.set_ylabel("Slowness relative to P (s/°)")
ax.set_title("PKiKP",fontweight="bold",fontsize=20)

savename = "VesPa_alignP_BH{}_{}_{}_{}_{}_{}_{}_{}s_{}_{}_{}.png".format(
                channel,stat,stack,n,smin,smax,plot_bsec,plot_esec,channel,freqmin,freqmax)

plt.savefig("./"+savename,dpi=240,bbox_inches='tight')
plt.cla()
plt.clf()
plt.close('all')
gc.collect()
