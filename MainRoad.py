# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 12:05:23 2025

@author: Maccarinelli
"""

# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit, prange
from numba_progress import ProgressBar
import numpy.typing as npt
from scipy.optimize import curve_fit
from scipy.stats import skewnorm
from pathlib import Path


from src.HBT_analysis import process_geetwo, calculate_photon_ratio_error, lorentzian, make_fit
from src.photon_generator import LightGenerator, QuantumDot, Detector, BeamSplitter, DeadTime, multi_stream_wrapper, Delay
from src.plot_utils import plotstream, arrival_distribution, statistics_test
from src.HanburyBrownTwiss import g2_experiment, g2_tdc
# from src.utils import gaussian



# %% Functions 





# You could probably increase speed by pre-creating the time_differences and temp_differences arrays, but it's fast enough when I use numba as-is.
@njit
def sparse_difference_matrix(stream1, stream2, max_difference, j_start=0):
    '''Gets all relevant differences between stream1 and stream2.
    Assumes both stream arrays are sorted!, loop is stopped when distance is greater than max_difference.
    Use same units for all inputs!
    
    Inputs:
        stream1 (nparray): 1D array containing photon emission times at output/detector 1
        stream2 (nparray): 1D array containing photon emission times at output/detector 2
        max_difference (float): time difference at which the differences are truncated

    Ouputs:
        time_differences (nparray): 1D array containing all time differences < max_difference between arrays
    '''
    # Alocate memory for output
    time_differences = np.empty(stream1.size**2, dtype=np.int64)
    # But we will not use the whole array, so keep track of where to end
    time_differences_idx = 0

    # Memory for the inner loop
    temp_differences = np.empty(stream2.size)

    # For the inner loop, the last useful time idx of stream2
    # j_start = 0

    for i in range(len(stream1)):

        for store_index, j in enumerate(range(j_start, len(stream2))):
            # Calc time differnces
            temp_differences[store_index] = stream2[j]-stream1[i]

            # Check if following calculations are useful
            if abs(temp_differences[store_index]) > max_difference: 
                if temp_differences[store_index] < 0:
                    j_start = j+1
                else:
                    break
        # Write to output array
        if time_differences_idx+store_index > time_differences.size:
            print("Time difference overflow")
            raise OverflowError("time_differences is too small, think about it you lazy sob")

        time_differences[time_differences_idx:time_differences_idx+store_index] = temp_differences[:store_index]
        # Update index of output array
        time_differences_idx += store_index
    if time_differences_idx >= time_differences.size:
        raise OverflowError("Trying to read higher idx than exists!")
        # time_differences_idx = time_differences.size

    # Return only the indices we wrote to
    return time_differences[:time_differences_idx], j_start

@njit(fastmath=True, parallel=True, nogil=True)
def sparse_coincidence_counts_jit(stream1:npt.NDArray[np.int64], stream2:npt.NDArray[np.int64], max_difference:int, chunk_size=10000, step_size = 100000, pbar=ProgressBar()):
    pbar.update(1)

    N_bins = int(2*np.ceil(max_difference/step_size)+1)
    N_chunks = int(np.ceil(stream1.size/chunk_size))
    hist = np.zeros(N_bins, dtype=np.int64)
    pbar.update(1)
    j_start=0


    for i in prange(N_chunks):
        chunk_start_idx = i*chunk_size
        chunk_stop_idx = min(stream1.size, chunk_start_idx+chunk_size)
        time_differences, j_start = sparse_difference_matrix(stream1[chunk_start_idx:chunk_stop_idx], stream2, max_difference, j_start)
        hist += np.histogram(time_differences, bins=N_bins, range=(-max_difference-0.5*step_size, max_difference+0.5*step_size))[0]
        pbar.update(1)
    
    return hist

def get_coincidence_counts_from_stream(stream1, stream2, max_difference, chunk_size, step_size):
    with ProgressBar(total=int(np.ceil(stream1.size/chunk_size))+2) as pbar:
        coincidence_counts = sparse_coincidence_counts_jit(stream1.astype(np.int64), stream2.astype(np.int64), max_difference, chunk_size, step_size, pbar)
    bins=np.arange(-max_difference, max_difference+step_size, step_size)
    
    return coincidence_counts, bins

def get_coincidence_counts_from_files(
        fname1,
        filename1,
        startstop_clock_ps = 15200,
        maxtime_ps =  int(0.001e12), # ps,
        stepsize_ps = 20000,
        chunk_size = 10000, # no. hits in a chunk
        ):
    
    tags1 = np.fromfile(fname1, dtype=np.uint64, sep="")
    tags2 = np.fromfile(filename1, dtype=np.uint64, sep="")
    
    shaped_tags1 = tags1.reshape((-1, 2))
    shaped_tags2 = tags2.reshape((-1, 2))
    
    shaped_tags1[::,1] *= startstop_clock_ps
    shaped_tags2[::,1] *= startstop_clock_ps
    
    timetags_ps1 = shaped_tags1[:, 1]  # Extract only timestamps
    timetags_ps2 = shaped_tags2[:, 1]   

    del shaped_tags1, shaped_tags2
    return get_coincidence_counts_from_stream(timetags_ps1, timetags_ps2, maxtime_ps, chunk_size=chunk_size, step_size=stepsize_ps)


def normalize(filename1, filename2, Counts_Coincident, acq_Time, startstop_clock_ps = 15200):
    """
    This function will act on the resulting objects of the function get_coincidence_counts_from_files() !!
    Keep in mind that for the functioning process of this function you need also to have in hand the names of 
    the files that you analised before !
    
    """
    
    tags1 = np.fromfile(filename1, dtype=np.uint64, sep="")
    tags2 = np.fromfile(filename2, dtype=np.uint64, sep="")
    
    shaped_tags1 = tags1.reshape((-1, 2))
    shaped_tags2 = tags2.reshape((-1, 2))
    
    shaped_tags1[::,1] *= startstop_clock_ps
    shaped_tags2[::,1] *= startstop_clock_ps
    
    timetags_ps1 = shaped_tags1[:, 1]  # Extract only timestamps
    timetags_ps2 = shaped_tags2[:, 1]   

    del shaped_tags1, shaped_tags2
    
    # print(r'The length of the timetags array is f(len(timetags_ps1))f, f(len(timetags)')
    
    rate1 = len(timetags_ps1) / (acq_Time * 1e12)
    rate2 = len(timetags_ps2) / (acq_Time * 1e12 )
    
    normalizedcounts = Counts_Coincident / (rate1 * rate2)
    
    return normalizedcounts
    
    

# %% Main Part !

PATH = Path("RAW")
folder_path = PATH
file_list = [f.name for f in folder_path.iterdir() if f.is_file() and f.suffix == '.bin']
names = [file_list[0], file_list[1]]

fname_detector2 = PATH / names[1]
fname_detector3 = PATH / names[0]

coincidence_counts, taus = get_coincidence_counts_from_files(
    fname_detector2, fname_detector3,
    stepsize_ps = 1,
    maxtime_ps=150000,
    chunk_size=17450
)



plt.figure()  # <- This line ensures a new figure for each iteration
plt.scatter(taus, coincidence_counts, s=0.8)
plt.xlabel(r"$\Delta \tau$ [ps]")
plt.ylabel("Counts")
plt.show()

Ncounts = normalize(fname_detector2, fname_detector3, coincidence_counts, 15)

plt.figure()  # <- This line ensures a new figure for each iteration
plt.scatter(taus, Ncounts, s=0.8)
plt.xlabel(r"$\Delta \tau$ [ps]")
plt.ylabel("Counts")
plt.show()