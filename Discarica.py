# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 15:54:31 2025
Ripostiglio di ProjectV1
@author: Maccarinelli
"""


# %% Fancy Creations from AI

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
    
