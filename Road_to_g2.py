# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 19:09:18 2025

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


PATH = Path("RAW/")



# Specify the folder path where your raw data is stored
folder_path = Path(PATH)
file_list = [f.name for f in folder_path.iterdir() if f.is_file() and f.suffix == '.bin'] 

pulse_lengths = [-10, -10, -15, -15, -20, -20, -25, -25, -25, -25, -35, -35, -40, -40,-40, -40, -45, -45, -50, -50, -55, -55, 
                  -5, -5, -60, -60, -61, -61, -62, -62, -65, -65, -70, -70, -75, -75, -75, -75, -75, -75, -75, -76, -76, -77, -77, -80, -80, 0, 0, 0, 0]
#pulse_lengths = [-60, -60]

# Combine them into a 2D NumPy array (first column: filenames, second column: pulse lengths)
result_array = np.array(list(zip(file_list, pulse_lengths)))










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
        
        # Construct the relative path to the file
        file_path = PATH / names

        bins, counts = hbt_histogram_from_file3(filename = file_path ,
                                                
                                               clock_ps=15200,
                                               bin_width_ps=B_width,
                                               max_tau_ps=M_Tau,
                                               SHFT = pulse_length, VIEW=SEE)

        results.append([bins, counts])  # Store the result

    return np.array(results, dtype=object)  # Use dtype=object for arrays of variable length


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

Visual= True
#Visual = False

match = find_eligible_files(result_array, -45)

results = []
results = process_matching_rows(match, 1000, 1530000, mode="odd", SEE = Visual)



#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

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


