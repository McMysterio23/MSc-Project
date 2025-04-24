# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:28:59 2025

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


# %% FILE PATHS
RAW_DATA_PATH_OFFICE = "/Users/Maccarinelli/Desktop/RAW/"
RAW_DATA_PATH_LAPTOP = "/Users/andreamaccarinelli/Desktop/LION/RAW_DATA/"


# Specify the folder path where your raw data is stored
# folder_path = Path(RAW_DATA_PATH_OFFICE)
folder_path = Path(RAW_DATA_PATH_LAPTOP)
file_list = [f.name for f in folder_path.iterdir() if f.is_file() and f.suffix == '.bin'] 

pulse_lengths = [-10, -10]

# Combine them into a 2D NumPy array (first column: filenames, second column: pulse lengths)
result_array = np.array(list(zip(file_list, pulse_lengths)))





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


def hbt_histogram_from_file(filename,
                            clock_ps=15200,
                            bin_width_ps=500,
                            max_tau_ps=100000,
                            VIEW = False,
                            SHFT = 0):
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

def hbt_histogram_from_file2(filename,
                            clock_ps=15200,
                            bin_width_ps=500,
                            max_tau_ps=100000,
                            VIEW=False,
                            SHFT=0):
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.fromfile(filename, dtype=np.uint64)
    data = data.reshape(-1, 2)
    
    # Convert Δt to picoseconds
    delta_t_ps = data[:, 1] * clock_ps

    # Filter by max_tau_ps (optional)
    delta_t_ps = delta_t_ps[delta_t_ps < max_tau_ps]

    # Build histogram directly from the delay times
    bins = np.arange(0, max_tau_ps, bin_width_ps)
    counts, _ = np.histogram(delta_t_ps, bins=bins)
    
    if VIEW:
        plt.figure()
        bin_centers = bins[:-1] + bin_width_ps / 2
        plt.plot(bin_centers, counts, label="Histogram", color='blue')
        plt.scatter(bin_centers, counts, s=3, color='darkorange', label="Data points")
        plt.xlabel("Δt [ps]")
        plt.ylabel("Counts")
        plt.title(f"HBT Histogram (Start-Stop Δt) for pulse length {SHFT} ps")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    return bins, counts


def hbt_histogram_from_file3(filename,
                            clock_ps=15200,
                            bin_width_ps=500,
                            max_tau_ps=100000,
                            VIEW=False,
                            SHFT=0):
    
    """
    Wolfgang Loeffler here helped so much !!
    """
    

    print("filename:", filename)

    data = np.fromfile(filename, dtype=np.uint64)
    data = data.reshape(-1, 2)
    print(filename)
    # Convert index to picoseconds
    # index_ps = data[:, 1] * clock_ps
    # data = [] #WL erase stuff

    # Filter by max_tau_ps (optional)
    # delta_t_ps = delta_t_ps[delta_t_ps < max_tau_ps]

    # Build histogram directly from the delay times
    bins = np.arange(0, 12000, 12)
    # counts, _ = np.histogram(data[:,0], bins=10, range=(0,12000))

    plt.hist(data[:,0], bins=1000, range=(0,12000))
    plt.xlim([4600,5000])
    plt.show()

    
    # if VIEW:
    #     plt.figure()
    #     bin_centers = bins[:-1] + bin_width_ps / 2
    #     plt.plot(bin_centers, counts, label="Histogram", color='blue')
    #     plt.scatter(bin_centers, counts, s=3, color='darkorange', label="Data points")
    #     plt.xlabel("Δt [ps]")
    #     plt.ylabel("Counts")
    #     plt.title(f"HBT Histogram (Start-Stop Δt) for pulse length {SHFT} ps")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()
        
    return bins, data[:, 0]


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
        
        

# Gaussian function using FWHM
def gaussian(x, A, b, fwhm, d):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return A * np.exp(-0.5 * ((x - d) / sigma) ** 2) + b

# Optional skewed gaussian if needed
def skewed_gaussian(x, a, loc, scale, amplitude):
    return amplitude * skewnorm.pdf(x, a, loc, scale)

def Do_Gauss_Fit(bins, counts, PrintParams = False):
    
    """
    This function will do a gaussian fit of an Histogram,
    starting from an array corresponding to the result of 
    a np.hist call and another array, called counts, that 
    is just the populations of each array.
    
    If you want to print out the Pandas Data Frame of all the parameters of the fit, pass in also as third
    argument a True valued variable !!!!!!!!!!
    
    
    Suggested use : Right after using the function hbt_histogram_from_file2()
    
    
    """
    
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Initial guess
    A0 = counts.max() - counts.min()
    b0 = counts.min()
    fwhm0 = (bins[-1] - bins[0]) / 10
    d0 = bin_centers[np.argmax(counts)]
    guess = [A0, b0, abs(fwhm0), d0]

    # Bounds
    lower_bounds = [0, 0, 1e-3, bins[0]]
    upper_bounds = [np.inf, np.inf, bins[-1] - bins[0], bins[-1]]

    try:
        popt, pcov = curve_fit(
            gaussian, bin_centers, counts, p0=guess,
            bounds=(lower_bounds, upper_bounds)
        )
    except RuntimeError as e:
        print("Fit failed:", e)
        return

    # Unpack fitted params
    A, b, fwhm, d = popt

    # Compute uncertainties (1σ standard deviation)
    perr = np.sqrt(np.diag(pcov))
    A_err, b_err, fwhm_err, d_err = perr

    # Plot
    plt.plot(bin_centers, counts, label='Histogram')
    plt.plot(bin_centers, gaussian(bin_centers, *popt), '--', label='Gaussian Fit')

    eqn_str = (
        f"$f(x) = A e^{{-0.5((x - d)/\\sigma)^2}} + b$\n"
        f"$A$ = {A:.2f} ± {A_err:.2f}, $b$ = {b:.2f} ± {b_err:.2f}\n"
        f"$\\mathrm{{FWHM}}$ = {fwhm:.2f} ± {fwhm_err:.2f} ps, $d$ = {d:.2f} ± {d_err:.2f} ps"
    )

    plt.xlabel("Δt [ps]")
    plt.ylabel("Counts")
    plt.title("Gaussian Fit to Histogram")
    plt.legend(title=eqn_str, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

    # Print with uncertainties
    print("Fitted parameters (±1σ):")
    print(f"Amplitude A     : {A:.2f} ± {A_err:.2f}")
    print(f"Offset b        : {b:.2f} ± {b_err:.2f}")
    print(f"FWHM            : {fwhm:.2f} ± {fwhm_err:.2f} ps")
    print(f"Center d        : {d:.2f} ± {d_err:.2f} ps")
    
    
    
    if (PrintParams):
        
        Names = ['A', 'b', 'FWHM', 'd']
        df = pd.DataFrame({
        'Parameter': Names,
        'Value': popt,
        'Uncertainty': perr
        })
        
        
        for i in range(4):
            print(f"The value of {exit.loc[i, 'Parameter']} as computed from the fit appears to be {exit.loc[i, 'Value']:.2f} ± {exit.loc[i, 'Uncertainty']:.2f}")
    
        return df
    
        



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

        bins, counts = hbt_histogram_from_file3(filename = PATH + names ,
                                               clock_ps=15200,
                                               bin_width_ps=B_width,
                                               max_tau_ps=M_Tau,
                                               SHFT = pulse_length, VIEW=SEE)

        results.append([bins, counts])  # Store the result

    return np.array(results, dtype=object)  # Use dtype=object for arrays of variable length


# %% SET POSITION OF WORKPLACE

'''
#ARE YOU IN THE OFFICE OR AT HOME ?

SET THE VARIABLE LOCATION IN THE CORRECT WAY !!!

FOR LOCATION == 1 ----> I AM CURRENTLY IN MY OFFICE WORKING
FOR LOCATION == 0 ----> I AM CURRENTLY WORKING FROM HOME 


'''

# LOCATION = 1
LOCATION = 0


if (LOCATION==1):
        
            PATH=RAW_DATA_PATH_OFFICE
else:  
            PATH = RAW_DATA_PATH_LAPTOP
            

# %%Execution of the Mio's Version of the code

#Ora finalmente funzionaaaaaaaa !

coincidence_counts, taus = get_coincidence_counts_from_files(
    PATH+file_list[1], PATH+file_list[0],
    stepsize_ps=4000,
    maxtime_ps=100000
)

plt.figure()  # <- This line ensures a new figure for each iteration
plt.plot(taus, coincidence_counts)
plt.xlabel("Δt [ps]")
plt.ylabel("Counts")
# plt.title(f"HBT Histogram (successive Δt) for pulse length {SHFT} ps")
plt.show()

# #coincidence_counts, taus, chunktimes = get_coincidence_counts_from_files("C:\Users\Maccarinelli\Desktop\RAW_DATA\EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-03-26T11_50_30.bin", "C:\Users\Maccarinelli\Desktop\RAW_DATA\EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T11_50_30.bin", stepsize_ps=1000, maxtime_ps=1000000000)





# %% FURTHER 
Visual= True
#Visual = False

#bins, counts = hbt_histogram_from_file(PATH+Zerops_C3, clock_ps=15200, bin_width_ps= 6500, max_tau_ps=150000, VIEW=Visual, SHFT=)  

#check_poisson_from_hist(counts)

#do_skew_gauss_fit(bins, counts)


# %% Further developments 

Visual= True
#Visual = False

match = find_eligible_files(result_array, -45)

results = []
results = process_matching_rows(match, 1000, 1530000, mode="odd", SEE = Visual)




# %% Sviluppi ?



data = np.fromfile(filename, dtype=np.uint64)
data = data.reshape(-1, 2)
pippo = np.arange(0,12000,12)
counts, _ = np.histogram(data[:,0], bins=pippo)
exitParams = Do_Gauss_Fit(pippo, counts, True)


# Normalize to get g2(tau)
bin_centers = pippo[:-1] + 12 / 2
baseline = np.mean(counts[(bin_centers > exitParams.loc[1,'Value'] * 12000)])  # adjust threshold if needed


g2_tau = counts / baseline

bin_centers = (pippo[:-1] + pippo[1:]) / 2
threshold = 0.8 * bin_centers.max()
mask = bin_centers > threshold
print(f"Selected {mask.sum()} bins for baseline averaging.")
baseline = np.mean(counts[mask])
g2_tau = counts / baseline


plt.plot(bin_centers, g2_tau, label='g₂(τ)')
plt.xlabel('τ [ps]')
plt.ylabel('g₂(τ)')
plt.title('Normalized Second-Order Correlation')
plt.grid(True)
plt.legend()
plt.show()


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


# %% Test on the data files to understand more...


def inspect_bin_file(filename, clock_ps=15200, num_lines=15):
    data = np.fromfile(filename, dtype=np.uint64)
    
    # Let's try to interpret it as pairs (start index, timestamp)
    if data.size % 2 != 0:
        print("⚠️ Warning: File doesn't contain an even number of entries. Might not be start-stop pairs.")
    
    reshaped = data[:2*num_lines].reshape(-1, 2)
    
    print(f"First {num_lines} entries in '{filename}':")
    for i, (idx, timestamp) in enumerate(reshaped):
        print(f"Line {i+1}: Index={idx}, Time={timestamp} → {timestamp * clock_ps} ps")
        
        
        

inspect_bin_file(PATH+match[1][0])