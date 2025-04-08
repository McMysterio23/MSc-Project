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



def get_coincidence_counts_from_files_partial(
        fname1,
        fname2,
        percentage=100,
        startstop_clock_ps=15200,
        maxtime_ps=int(0.001e12),  # ps
        stepsize_ps=20000,
        chunk_size=10000  # no. hits in a chunk
):
    """
    Reads timestamp data from two binary files and processes only a given percentage of the data.

    Args:
    - fname1 (str): Path to the first binary file.
    - fname2 (str): Path to the second binary file.
    - percentage (float): Percentage of data to analyze (0-100).
    - startstop_clock_ps (int): Clock period in picoseconds.
    - maxtime_ps (int): Maximum time window in picoseconds.
    - stepsize_ps (int): Step size for coincidence analysis.
    - chunk_size (int): Number of hits per chunk.

    Returns:
    - Coincidence counts from the selected portion of the data.
    """

    # Read full data from both files
    tags1 = np.fromfile(fname1, dtype=np.uint64, sep="")
    tags2 = np.fromfile(fname2, dtype=np.uint64, sep="")

    # Reshape into two columns
    shaped_tags1 = tags1.reshape((-1, 2))
    shaped_tags2 = tags2.reshape((-1, 2))

    # Determine the number of rows to keep based on percentage
    num_rows1 = int(shaped_tags1.shape[0] * (percentage / 100))
    num_rows2 = int(shaped_tags2.shape[0] * (percentage / 100))

    # Select only the specified percentage of rows
    shaped_tags1 = shaped_tags1[:num_rows1]
    shaped_tags2 = shaped_tags2[:num_rows2]

    # Convert timestamps to picoseconds
    shaped_tags1[:, 1] *= startstop_clock_ps
    shaped_tags2[:, 1] *= startstop_clock_ps

    # Extract only timestamps
    timetags_ps1 = shaped_tags1[:, 1]
    timetags_ps2 = shaped_tags2[:, 1]

    # Cleanup unused arrays
    del shaped_tags1, shaped_tags2

    return get_coincidence_counts_from_stream(timetags_ps1, timetags_ps2, maxtime_ps, chunk_size=chunk_size, step_size=stepsize_ps)



def Get_Data(
        Fname1,
        startstop_clock_ps = 15200,maxtime_ps =  int(0.001e12), # ps,
        stepsize_ps = 20000,
        chunk_size = 10000, # no. hits in a chunk
        
        ):
    #Reading from the file
    tags = np.fromfile(Fname1, dtype=np.uint64, sep="")
    
   # Operations on the arrays  
    shaped_tags = tags.reshape((-1, 2))
    shaped_tags[::,1] *= startstop_clock_ps
    
    #Extract Timetags
    timetags_ps = shaped_tags[:, 1]
    
   

    plt.plot(timetags_ps, marker="o", linestyle="none")  # Scatter plot of timestamps
    plt.xlabel("Event Number")
    plt.ylabel("Time (ps)")
    plt.title("Photon Detection Times for Detector 1")
    plt.show()

    
    #del shaped_tags
    return timetags_ps
    
"""
def process_snsdp_data(file_path, include_start_index=False):
    
    Process SNSPD data from a binary file, extract timestamps and optional start indices,
    and compute time differences between photon detection events.

    Args:
    - file_path (str): Path to the binary file containing SNSPD data.
    - include_start_index (bool): Whether the start index is included in the file (default: False).
    
    Returns:
    - timestamps (np.ndarray): Array of timestamps (in picoseconds).
    - start_indices (np.ndarray, optional): Array of start indices (if included).
    - time_differences (np.ndarray): Array of time differences between consecutive timestamps.
    
    
    # Read the binary file data
    data = np.fromfile(file_path, dtype=np.uint64)
    
    if include_start_index:
        # If start index is included, reshape into two columns: timestamp and start index
        data = data.reshape(-1, 2)
        timestamps = data[:, 0]  # Extract timestamps
        start_indices = data[:, 1]  # Extract start indices (if included)
    else:
        timestamps = data  # Only timestamps, no start index
        
   # Filter out zero timestamps (or any invalid ones)
    valid_timestamps = timestamps[timestamps > 0]  # Only keep timestamps greater than 0
    
    # Calculate time differences between consecutive valid timestamps
    time_differences = np.diff(valid_timestamps)
    
    # Plot the valid timestamps (only y-values, not the event number)
    plt.figure(figsize=(10, 6))
    plt.plot(valid_timestamps, marker='o', linestyle='none', color='blue')
    plt.xlabel('Event Number')
    plt.ylabel('Timestamp (ps)')
    plt.title('Photon Detection Timestamps (Valid Entries)')
    plt.grid(True)
    plt.show()
    
    if include_start_index:
        return timestamps, start_indices, time_differences
    else:
        return timestamps, time_differences
"""


def process_snsdp_data(file_path, include_start_index=False):
    """
    Process SNSPD data from a binary file, extract timestamps and optional start indices,
    and compute time differences between photon detection events.

    Args:
    - file_path (str): Path to the binary file containing SNSPD data.
    - include_start_index (bool): Whether the start index is included in the file (default: False).
    
    Returns:
    - timestamps (np.ndarray): Array of timestamps (in picoseconds).
    - start_indices (np.ndarray, optional): Array of start indices (if included).
    - time_differences (np.ndarray): Array of time differences between consecutive timestamps.
    """
    
    # Read the binary file data
    data = np.fromfile(file_path, dtype=np.uint64)
    
    if include_start_index:
        # If start index is included, reshape into two columns: timestamp and start index
        data = data.reshape(-1, 2)
        timestamps = data[:, 0]  # Extract timestamps
        start_indices = data[:, 1]  # Extract start indices (if included)
    else:
        timestamps = data  # Only timestamps, no start index
        
   # Filter out zero timestamps (or any invalid ones)
    valid_timestamps = timestamps[timestamps > 0]  # Only keep timestamps greater than 0
    
    # Calculate time differences between consecutive valid timestamps
    time_differences = np.diff(valid_timestamps)
    
    # Plot the valid timestamps (only y-values, not the event number)
    plt.figure(figsize=(10, 6))
    plt.plot(valid_timestamps, marker='o', linestyle='none', color='blue')
    plt.xlabel('Event Number')
    plt.ylabel('Timestamp (ps)')
    plt.title('Photon Detection Timestamps (Valid Entries)')
    plt.grid(True)
    plt.show()
    
    if include_start_index:
        return timestamps, start_indices, time_differences
    else:
        return timestamps, time_differences
    

# %%Execution of the Mio's Version of the code


coincidence_counts, taus, chunktimes = get_coincidence_counts_from_files(
    "C:/Users/Maccarinelli/Desktop/RAW_DATA/EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-03-26T11_50_30.bin",
    "C:/Users/Maccarinelli/Desktop/RAW_DATA/EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T11_50_30.bin",
    stepsize_ps=1000,
    maxtime_ps=1000000000
)
#coincidence_counts, taus, chunktimes = get_coincidence_counts_from_files("C:\Users\Maccarinelli\Desktop\RAW_DATA\EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-03-26T11_50_30.bin", "C:\Users\Maccarinelli\Desktop\RAW_DATA\EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T11_50_30.bin", stepsize_ps=1000, maxtime_ps=1000000000)



# %% Execution of my version of the code 

#Get_Data(RAW_DATA_PATH+Zerops_C3)


#timestamps, time_differences = process_snsdp_data(RAW_DATA_PATH+Zerops_C3, include_start_index=False)

'''
#ARE YOU IN THE OFFICE OR AT HOME ?

SET THE VARIABLE LOCATION IN THE CORRECT WAY !!!

FOR LOCATION == 1 ----> I AM CURRENTLY IN MY OFFICE WORKING
FOR LOCATION == 0 ----> I AM CURRENTLY WORKING FROM HOME 


'''

#LOCATION = 1
LOCATION = 0


if (LOCATION==1):
        
            PATH=RAW_DATA_PATH_OFFICE
else:  
            PATH = RAW_DATA_PATH_LAPTOP
            

coincidence_counts, taus = get_coincidence_counts_from_files_partial(
    PATH+"EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-03-26T11_50_30.bin",
    PATH+"EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T11_50_30.bin",
    percentage=40,
    stepsize_ps=1000,
    maxtime_ps=1000000000,
)




