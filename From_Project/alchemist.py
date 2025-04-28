# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 14:26:36 2025

@author: Maccarinelli
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar
import numpy.typing as npt
from pathlib import Path

PATH = Path("RAW")
folder_path = PATH
file_list = [f.name for f in folder_path.iterdir() if f.is_file() and f.suffix == '.bin']
names = [file_list[0], file_list[1], file_list[2], file_list[3]]

fname_detector2 = PATH / names[0]
fname_detector3 = PATH / names[1]

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
            break
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

# def get_coincidence_counts_from_stream(stream1, stream2, max_difference, chunk_size, step_size):
#     with ProgressBar(total=int(np.ceil(stream1.size/chunk_size))+2) as pbar:
#         coincidence_counts = sparse_coincidence_counts_jit(stream1.astype(np.int64), stream2.astype(np.int64), max_difference, chunk_size, step_size, pbar)
#     bins=np.arange(-max_difference, max_difference + step_size, step_size)
#     taus = (bins[:-1] + bins[1:]) / 2
#     return coincidence_counts, taus

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
    timetags_ps1 = shaped_tags1.sum(1)
    timetags_ps2 = shaped_tags2.sum(1)
    del shaped_tags1, shaped_tags2
    return get_coincidence_counts_from_stream(timetags_ps1, timetags_ps2, maxtime_ps, chunk_size=chunk_size, step_size=stepsize_ps)


# def normalize(filename1, filename2, Counts_Coincident, acq_Time, startstop_clock_ps=15200, step_size_ps=1):
#     """
#     Normalizes the raw coincidence counts into g2(τ).
#     """
#     tags1 = np.fromfile(filename1, dtype=np.uint64, sep="")
#     tags2 = np.fromfile(filename2, dtype=np.uint64, sep="")
    
#     shaped_tags1 = tags1.reshape((-1, 2))
#     shaped_tags2 = tags2.reshape((-1, 2))
    
#     shaped_tags1[:, 1] *= startstop_clock_ps
#     shaped_tags2[:, 1] *= startstop_clock_ps
    
#     timetags_ps1 = shaped_tags1[:, 1]
#     timetags_ps2 = shaped_tags2[:, 1]
    
#     del shaped_tags1, shaped_tags2
    
#     # Photon rates [photons per second]
#     rate1 = len(timetags_ps1) / acq_Time
#     rate2 = len(timetags_ps2) / acq_Time

#     # Expected random coincidences per bin
#     expected_counts_per_bin = rate1 * rate2 * (step_size_ps * 1e-12) * acq_Time

#     normalized_counts = Counts_Coincident / expected_counts_per_bin
    
#     return normalized_counts



def normalize(filename1, filename2, Counts_Coincident, acq_Time, startstop_clock_ps=15200, step_size_ps=1, min_expected_counts=5):
    """
    Normalizes the raw coincidence counts into g2(τ).
    Includes protection against very small bin sizes.
    """
    # Read the time tags from the files
    tags1 = np.fromfile(filename1, dtype=np.uint64, sep="")
    tags2 = np.fromfile(filename2, dtype=np.uint64, sep="")
    
    # Reshape the tags into pairs (tag, timestamp)
    shaped_tags1 = tags1.reshape((-1, 2))
    shaped_tags2 = tags2.reshape((-1, 2))
    
    # Scale the timestamps
    shaped_tags1[:, 1] *= startstop_clock_ps
    shaped_tags2[:, 1] *= startstop_clock_ps
    
    timetags_ps1 = shaped_tags1[:, 1]
    timetags_ps2 = shaped_tags2[:, 1]
    
    del shaped_tags1, shaped_tags2  # Free memory
    
    # Photon rates [photons per second]
    rate1 = len(timetags_ps1) / acq_Time
    rate2 = len(timetags_ps2) / acq_Time
    
    print(f"Rate of acquisition for Detector 2 is {rate1}\n")
    print(f"Rate of acquisition for Detector 3 is {rate2}\n")

    # Calculate expected random coincidences per bin
    expected_counts_per_bin = rate1 * rate2 * (step_size_ps * 1e-12) * acq_Time
    
    # Debug: print expected counts per bin to verify values
    print(f"Expected counts per bin: {expected_counts_per_bin:.6e}")
    
    # Check if expected counts per bin are too small
    if expected_counts_per_bin < min_expected_counts:
        print(f"Warning: expected counts per bin ({expected_counts_per_bin:.2e}) is too low for reliable normalization.")
        print("Consider using a larger bin size or smoothing the data after normalization.")
    
    # Normalize counts
    normalized_counts = Counts_Coincident / expected_counts_per_bin
    
    return normalized_counts


STEP_ps = 4

coincidence_counts, taus = get_coincidence_counts_from_files(fname_detector2, fname_detector3, stepsize_ps=STEP_ps, maxtime_ps=15000)

print(f"current binsize : {STEP_ps}\n")

# UNCOMMENT IF YOU DESIRE TO LOOK AT THE HISTOGRAM BEFORE THE NORMALIZATION

# plt.figure()  # <- This line ensures a new figure for each iteration
# plt.scatter(taus, coincidence_counts, s=0.8)
# plt.xlabel(r"$\Delta \tau$ [ps]")
# plt.ylabel("Counts")
# plt.show()


Ncounts = normalize(
    fname_detector2, fname_detector3,
    coincidence_counts,
    15,  
    step_size_ps=STEP_ps  
)

plt.figure()
plt.title(f"Second Order Correlation Function for -10ps delay pulses and binsize {STEP_ps}ps")
plt.plot(taus, Ncounts, linewidth=0.8, color='#FF9B00', label = 'Line')  
plt.scatter(taus, Ncounts, s=4, color='black', label = 'Data Points')

# Axis labels bigger
plt.xlabel(r"$\Delta \tau$ [ps]", fontsize=16)
plt.ylabel(r"$g_{2}(\tau)$", fontsize=16)

# Increase tick size
plt.xticks(fontsize=10)
plt.yticks(fontsize=13)

# Legend (if you want it)
plt.legend(fontsize=14, loc='best', frameon=True)
plt.xlim((10250, +10675))
plt.ylim((-3, +150))
plt.tight_layout()
plt.show()
