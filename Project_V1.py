# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:28:59 2025

@author: Maccarinelli
"""



# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar
import numpy.typing as npt




# %% FILE PATHS
RAW_DATA_PATH_OFFICE = "C:/Users/Maccarinelli/Desktop/RAW_DATA/"
RAW_DATA_PATH_LAPTOP = "/Users/andreamaccarinelli/Desktop/LION/RAW_DATA/"


Zerops_C2 = "EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-03-26T11_50_30.bin"
Zerops_C3 = "EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T11_50_30.bin"






# %% Definition of the functions

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
        fname2,
        startstop_clock_ps = 15200,
        maxtime_ps =  int(0.001e12), # ps,
        stepsize_ps = 20000,
        chunk_size = 10000, # no. hits in a chunk
        ):
    
    tags1 = np.fromfile(fname1, dtype=np.uint64, sep="")
    tags2 = np.fromfile(fname2, dtype=np.uint64, sep="")
    
    shaped_tags1 = tags1.reshape((-1, 2))
    shaped_tags2 = tags2.reshape((-1, 2))
    
    shaped_tags1[::,1] *= startstop_clock_ps
    shaped_tags2[::,1] *= startstop_clock_ps
    
    timetags_ps1 = shaped_tags1[:, 1]  # Extract only timestamps
    timetags_ps2 = shaped_tags2[:, 1]   

    del shaped_tags1, shaped_tags2
    return get_coincidence_counts_from_stream(timetags_ps1, timetags_ps2, maxtime_ps, chunk_size=chunk_size, step_size=stepsize_ps)


def hbt_histogram_from_file(filename, clock_ps=15200, bin_width_ps=500, max_tau_ps=100000):
    data = np.fromfile(filename, dtype=np.uint64)
    data = data.reshape(-1, 2)
    timestamps = data[:,1] * clock_ps
    
    # Compute time differences between successive detections
    dt = np.diff(timestamps)
    dt = dt[dt < max_tau_ps]  # Remove huge delays
    
    bins = np.arange(0, max_tau_ps, bin_width_ps)
    counts, _ = np.histogram(dt, bins=bins)
    
    plt.plot(bins[:-1], counts)
    plt.xlabel("Δt [ps]")
    plt.ylabel("Counts")
    plt.title("HBT Histogram (successive Δt)")
    plt.show()

    return bins, counts

# %% SET POSITION OF WORKPLACE

'''
#ARE YOU IN THE OFFICE OR AT HOME ?

SET THE VARIABLE LOCATION IN THE CORRECT WAY !!!

FOR LOCATION == 1 ----> I AM CURRENTLY IN MY OFFICE WORKING
FOR LOCATION == 0 ----> I AM CURRENTLY WORKING FROM HOME 


'''

LOCATION = 1
#LOCATION = 0


if (LOCATION==1):
        
            PATH=RAW_DATA_PATH_OFFICE
else:  
            PATH = RAW_DATA_PATH_LAPTOP
            

# %%Execution of the Mio's Version of the code


coincidence_counts, taus, chunktimes = get_coincidence_counts_from_files(
    "C:/Users/Maccarinelli/Desktop/RAW_DATA/EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-03-26T11_50_30.bin",
    "C:/Users/Maccarinelli/Desktop/RAW_DATA/EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T11_50_30.bin",
    stepsize_ps=1000,
    maxtime_ps=1000000000
)
#coincidence_counts, taus, chunktimes = get_coincidence_counts_from_files("C:\Users\Maccarinelli\Desktop\RAW_DATA\EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-03-26T11_50_30.bin", "C:\Users\Maccarinelli\Desktop\RAW_DATA\EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T11_50_30.bin", stepsize_ps=1000, maxtime_ps=1000000000)




# %% FURTHER DEVELOPMENTS

hbt_histogram_from_file(PATH+Zerops_C2, clock_ps=15200, bin_width_ps=300, max_tau_ps=1000000)
