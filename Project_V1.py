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
from scipy.optimize import curve_fit
from scipy.stats import skewnorm
from pathlib import Path

from src.HBT_analysis import process_geetwo, calculate_photon_ratio_error, lorentzian, make_fit
from src.photon_generator import LightGenerator, QuantumDot, Detector, BeamSplitter, DeadTime, multi_stream_wrapper, Delay
from src.plot_utils import plotstream, arrival_distribution, statistics_test
from src.HanburyBrownTwiss import g2_experiment, g2_tdc
from src.utils import gaussian


# %% FILE PATHS
RAW_DATA_PATH_OFFICE = "/Users/Maccarinelli/Desktop/RAW/"
RAW_DATA_PATH_LAPTOP = "/Users/andreamaccarinelli/Desktop/LION/RAW_DATA/"


# Specify the folder path where your raw data is stored
folder_path = Path(RAW_DATA_PATH_OFFICE)
file_list = [f.name for f in folder_path.iterdir() if f.is_file() and f.suffix == '.bin'] 

pulse_lengths = [-10, -10, -15, -15, -20, -20, -25, -25, -25, -25, -35, -35, -40, -40,-40, -40, -45, -45, -50, -50, -55, -55, 
                 -5, -5, -60, -60, -61, -61, -62, -62, -65, -65, -70, -70, -75, -75, -75, -75, -75, -75, -75, -76, -76, -77, -77, -80, -80, 0, 0, 0, 0]

# Combine them into a 2D NumPy array (first column: filenames, second column: pulse lengths)
result_array = np.array(list(zip(file_list, pulse_lengths)))





Zerops_C2 = "EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-03-26T11_50_30.bin"
Zerops_C3 = "EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T11_50_30.bin"
Ten_C2 = "EOM_-10ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-03-26T11_50_30.bin"
Ten_C3 = "EOM_-10ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T11_50_30.bin"





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


def hbt_histogram_from_file(filename, clock_ps=15200, bin_width_ps=500, max_tau_ps=100000, VIEW = False, SHFT = 0):
    data = np.fromfile(filename, dtype=np.uint64)
    data = data.reshape(-1, 2)
    timestamps = data[:,1] * clock_ps
    
    # Compute time differences between successive detections
    dt = np.diff(timestamps)
    dt = dt[dt < max_tau_ps]  # Remove huge delays
    
    bins = np.arange(0, max_tau_ps, bin_width_ps)
    counts, _ = np.histogram(dt, bins=bins)
    
    if (VIEW):
        plt.figure()  # <- This line ensures a new figure for each iteration
        plt.plot(bins[:-1], counts)
        plt.xlabel("Δt [ps]")
        plt.ylabel("Counts")
        plt.title(f"HBT Histogram (successive Δt) for pulse length {SHFT} ps")
        plt.show()
        
    return bins, counts


def start_stop_histogram_from_file(filename, clock_ps=15200, bin_width_ps=1, max_tau_ps=15300, VIEW=False, SHFT=0):
    data = np.fromfile(filename, dtype=np.uint64)
    data = data.reshape(-1, 2)
    
    channels = data[:, 0]
    timestamps = data[:, 1] * clock_ps

    # Extract and sort start and stop timestamps
    t_start = np.sort(timestamps[channels == 1])
    t_stop = np.sort(timestamps[channels == 2])

    delays = []
    j = 0

    for t in t_start:
        while j < len(t_stop) and t_stop[j] < t:
            j += 1
        if j < len(t_stop):
            dt = t_stop[j] - t
            print("Raw Δt:", dt)
            if dt < max_tau_ps:
                delays.append(dt)

    delays = np.array(delays)
    
    bins = np.arange(0, max_tau_ps + bin_width_ps, bin_width_ps)
    counts, _ = np.histogram(delays, bins=bins)

    if VIEW:
        plt.figure()  # ensures one figure per histogram
        plt.plot(bins[:-1], counts)
        plt.xlabel("Δt [ps]")
        plt.ylabel("Counts")
        plt.title(f"Start-Stop HBT Histogram for pulse length {SHFT} ps")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return bins, counts



def check_poisson_from_hist(counts):
    mean = np.mean(counts)
    var = np.var(counts)
    print(f"Mean: {mean:.2f}, Variance: {var:.2f}")
    if abs(mean - var) < 0.1 * mean:
        print("✔️ Histogram shape is consistent with Poisson statistics.")
    else:
        print("❌ Histogram does not closely match Poisson distribution.")
        
        

def gaussian(x, A, b, hwhm, d):
    sigma = hwhm * 2 / (2*np.sqrt(2*np.log(2)))
    return A*np.exp(-0.5*((x-d)/sigma)**2) + b

def skewed_gaussian(x, a, loc, scale, amplitude):
    return amplitude * skewnorm.pdf(x, a, loc, scale)


def Do_Gauss_Fit(bins, counts):
    
    
    """
    # Let's assume you've got bins and counts from your histogram
    # from: bins, counts = hbt_histogram_from_file(...)
    """
    # Let's assume you've got bins and counts from your histogram
    # from: bins, counts = hbt_histogram_from_file(...)
    
    # Compute bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Guess some initial parameters [A, b, hwhm, d]
    guess = [counts.max(), counts.min(), (bins[-1] - bins[0])/10, bin_centers[np.argmax(counts)]]
    
    # Fit the histogram to your gaussian model
    popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=guess)
    
    # Plot result
    plt.plot(bin_centers, counts, label='Histogram')
    plt.plot(bin_centers, gaussian(bin_centers, *popt), label='Gaussian fit', linestyle='--')
    plt.xlabel("Δt [ps]")
    plt.ylabel("Counts")
    plt.title("Gaussian Fit to Histogram")
    plt.legend()
    plt.show()
    
    # Print fitted parameters
    print("Fitted parameters:")
    print(f"Amplitude A: {popt[0]:.2f}")
    print(f"Offset b: {popt[1]:.2f}")
    print(f"HWHM: {popt[2]:.2f} ps")
    print(f"Center d: {popt[3]:.2f} ps")

def do_skew_gauss_fit(bins, counts):
    
    """
    To be used after using hbt_histogram_from_file() in order to have bins and counts to use as arguments for this function
    
    """
    
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Get center of each bin
    
    # Initial guess: skewness, mean, stddev, amplitude
    p0 = [0, bin_centers[np.argmax(counts)], np.std(bin_centers), max(counts)]
    
    # Fit
    params, _ = curve_fit(skewed_gaussian, bin_centers, counts, p0=p0)
    
    # Plot
    plt.plot(bin_centers, counts, label='Data')
    plt.plot(bin_centers, skewed_gaussian(bin_centers, *params), label='Skewed Gaussian fit', linestyle='--')
    plt.xlabel("Δt [ps]")
    plt.ylabel("Counts")
    plt.title("HBT Histogram with Skewed Gaussian Fit")
    plt.legend()
    # Format the text for the annotation
    a, loc, scale, amplitude = params
    textstr = '\n'.join((r'$a=%.2f$' % (a, ),r'$\mu=%.2f$ ps' % (loc, ), r'$\sigma=%.2f$ ps' % (scale, ), r'$A=%d$' % (amplitude, )))

    # Add the text box to the plot
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    plt.show()
    
    # Print fitted parameters
    
    print(f"Fitted skewness: {a:.2f}, mean: {loc:.2f}, std dev: {scale:.2f}, amplitude: {amplitude:.0f}")




def find_eligible_files(DataArray, Pulse_Length):
    """
    Finds given the DATA ARRAY, and the pulse length desired the matching rows and gives out the desired data array
    """
    
    # Define the pulse length value you want to search for
    target_pulse_length = Pulse_Length
    
    # Ensure the second column is converted to integers (assuming it contains numeric strings)
    second_column_as_int = DataArray[:, 1].astype(int)
    
    # Use np.where to find the indices of rows where the second column matches the target value
    indices = np.where(second_column_as_int == target_pulse_length)[0]
    
    # Extract the rows using the indices
    matching_rows = DataArray[indices]
    
    # Check if there are matching rows and print them
    if matching_rows.size > 0:
        print(f"There are {len(matching_rows[:, 0])} rows matching the pulse length {target_pulse_length}:")
        print(matching_rows)
    else:
        print(f"No rows found with pulse length {target_pulse_length}")
    
    return matching_rows

def process_matching_rows(matching_rows, B_width, M_Tau, mode="even", SEE = False):
    """
    Iterates through even or odd rows of matching_rows,
    calls hbt_histogram_from_file on each, and stores [bins, counts] results.
    
    Returns:
        A numpy array (dtype=object) where each row is [bins, counts].
    """
    if mode not in ("even", "odd"):
        raise ValueError("mode must be either 'even' or 'odd'")

    start_index = 0 if mode == "even" else 1

    results = []  # Collect results here

    for i in range(start_index, len(matching_rows), 2):
        names = matching_rows[i, 0]
        pulse_length = int(matching_rows[i, 1])

        bins, counts = hbt_histogram_from_file(filename = RAW_DATA_PATH_OFFICE + names , clock_ps=15200, bin_width_ps=B_width, max_tau_ps=M_Tau, SHFT = pulse_length, VIEW=SEE)

        results.append([bins, counts])  # Store the result

    return np.array(results, dtype=object)  # Use dtype=object for arrays of variable length


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


coincidence_counts, taus = get_coincidence_counts_from_files(
    "C:/Users/Maccarinelli/Desktop/RAW/EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T11_50_30.bin",
    "C:/Users/Maccarinelli/Desktop/RAW/EOM_-15ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T11_54_37.bin",
    stepsize_ps=4000,
    maxtime_ps=100000
)

plt.figure()  # <- This line ensures a new figure for each iteration
plt.plot(taus, coincidence_counts)
plt.xlabel("Δt [ps]")
plt.ylabel("Counts")
#plt.title(f"HBT Histogram (successive Δt) for pulse length {SHFT} ps")
plt.show()

#coincidence_counts, taus, chunktimes = get_coincidence_counts_from_files("C:\Users\Maccarinelli\Desktop\RAW_DATA\EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-03-26T11_50_30.bin", "C:\Users\Maccarinelli\Desktop\RAW_DATA\EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T11_50_30.bin", stepsize_ps=1000, maxtime_ps=1000000000)




# %% FURTHER DEVELOPMENTS
Visual= True
#Visual = False

#bins, counts = hbt_histogram_from_file(PATH+Zerops_C3, clock_ps=15200, bin_width_ps= 6500, max_tau_ps=150000, VIEW=Visual, SHFT=)  

#check_poisson_from_hist(counts)

#do_skew_gauss_fit(bins, counts)


match = find_eligible_files(result_array, 0)

results = []
results = process_matching_rows(match, 15000, 1530000, mode="even", SEE = Visual)


# %%


data = np.fromfile('C:/Users/Maccarinelli/Desktop/RAW_DATA/EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-03-26T11_50_30.bin', dtype=np.uint64)

# Check the shape of the data
print(f"Data shape: {data.shape}")

# If you expect two columns, reshape accordingly (if it was stored as pairs of values)
# This assumes the binary data has timestamps and start index paired together
data = data.reshape(-1, 2)

# Inspect the first few rows
print("First few rows of data:")
print(data[:10])  # Displaying first 10 rows


# %% Next

Visual= True
#Visual = False

# Assuming the filename is correct and the file structure is as described
def load_timestamps_from_file(filename):
    """
    Loads timestamps from a binary file.

    Assumes the binary file contains a flat array of 64-bit unsigned integers (timestamps in picoseconds).
    
    Returns:
        np.ndarray: 1D array of timestamps.
    """
    try:
        timestamps = np.fromfile(filename, dtype=np.uint64)
        return timestamps
    except Exception as e:
        print(f"Failed to load binary data from {filename}: {e}")
        return np.array([])  # Return empty array to avoid crashes




from numba import njit
from numba_progress import ProgressBar
import numpy as np

from numba import njit, prange
from numba_progress import ProgressBar
import numpy as np

from numba import njit, prange
from numba_progress import ProgressBar
import numpy as np

def update_progress_bar(pbar, start_idx, end_idx):
    """
    Update the progress bar manually outside the numba function.
    """
    pbar.update(end_idx - start_idx)

@njit(fastmath=True, nogil=True)
def compute_autocorr_chunk(timestamps, start_idx, end_idx, bin_width_ps, max_tau_ps, counts):
    """
    Compute autocorrelation for a chunk of timestamps in parallel.
    
    Parameters:
        timestamps (np.ndarray): Sorted 1D array of timestamps in picoseconds.
        start_idx (int): Starting index for processing.
        end_idx (int): Ending index for processing.
        bin_width_ps (int): Width of histogram bins.
        max_tau_ps (int): Maximum correlation time (τ) in picoseconds.
        counts (np.ndarray): Array to accumulate counts.
    """
    num_bins = max_tau_ps // bin_width_ps
    for i in range(start_idx, end_idx):
        t0 = timestamps[i]
        for j in range(i + 1, len(timestamps)):
            dt = timestamps[j] - t0
            if dt >= max_tau_ps:
                break
            bin_index = dt // bin_width_ps
            counts[bin_index] += 1

def start_stop_autocorr_histogram_numba(timestamps, bin_width_ps, max_tau_ps, chunk_size=100_000):
    """
    Fast autocorrelation computation with manual progress updates in parallel.

    Parameters:
        timestamps (np.ndarray): Sorted timestamps.
        bin_width_ps (int): Width of histogram bins.
        max_tau_ps (int): Max τ in picoseconds.
        chunk_size (int): How many timestamps to process at once.

    Returns:
        bins (np.ndarray), counts (np.ndarray)
    """
    num_bins = max_tau_ps // bin_width_ps
    counts = np.zeros(num_bins, dtype=np.uint32)
    n = len(timestamps)
    
    # Create the progress bar here (outside the numba function)
    pbar = ProgressBar(total=n)
    
    for start_idx in prange(0, n, chunk_size):
        end_idx = min(start_idx + chunk_size, n)
        compute_autocorr_chunk(timestamps, start_idx, end_idx, bin_width_ps, max_tau_ps, counts)
        
        # Update the progress bar manually
        update_progress_bar(pbar, start_idx, end_idx)

    bins = np.arange(num_bins) * bin_width_ps + bin_width_ps // 2
    return bins, counts


def process_matching_rows_by_mode(matching_rows, B_width, M_Tau, mode="even", SEE=False):
    """
    Processes even or odd rows of matching_rows, extracts timestamps,
    and computes autocorrelation histograms for each file using Numba acceleration.
    
    Parameters:
        matching_rows (np.ndarray): Each row contains [filename, ...]
        B_width (int): Bin width in picoseconds
        M_Tau (int): Max Δt (ps) to consider
        mode (str): Selects "even" or "odd" rows to process
        SEE (bool): Whether to show plots

    Returns:
        np.ndarray: A numpy array (dtype=object) where each row is [bins, counts].
    """
    if mode not in ("even", "odd"):
        raise ValueError("mode must be either 'even' or 'odd'")

    start_index = 0 if mode == "even" else 1
    results = []

    for i in range(start_index, len(matching_rows), 2):
        filename = PATH + matching_rows[i, 0]
        print(f"\nProcessing file: {filename}")

        try:
            timestamps = load_timestamps_from_file(filename)

            # Sort timestamps for autocorrelation logic to work
            timestamps = np.sort(timestamps)

            # Use fast, memory-efficient version
            bins, counts = start_stop_autocorr_histogram_numba(timestamps, B_width, M_Tau)

            if SEE:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.bar(bins, counts, width=B_width, align="center")
                plt.xlabel("Δt (ps)")
                plt.ylabel("Counts")
                plt.title(f"Histogram for file: {filename}")
                plt.tight_layout()
                plt.show()

            results.append([bins, counts])

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

    return np.array(results, dtype=object)



match = find_eligible_files(result_array, -15)

results = []
results = process_matching_rows_by_mode(match, 1, 15300, mode="even", SEE = Visual)
