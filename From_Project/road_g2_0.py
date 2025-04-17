# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 16:50:32 2025

@author: Maccarinelli
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit, prange
from numba_progress import ProgressBar
import numpy.typing as npt
from scipy.optimize import curve_fit
from pathlib import Path
from src.Librerie import Do_Gauss_Fit, gaussian
from src.utils import peakfinder

# Define the Gaussian function
def gaussian2(x, A, mu, sigma, d):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + d

# File loading and data preparation
PATH = Path("RAW")
filename = PATH / 'EOM_-40ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T12_04_38.bin'

# Load data
data = np.fromfile(filename, dtype=np.uint64)
data = data.reshape(-1, 2)

# Define the bins for the histogram
pippo = np.arange(0, 12000, 12)  # Array of the right-sided locations of the bins

# Create the histogram by subdividing the timestamps on the bins
counts, _ = np.histogram(data[:, 0], bins=pippo)

# Calculate bin centers (the x positions of the histogram)
bin_centers = (pippo[:-1] + pippo[1:]) / 2

# Peak detection using peakfinder
peaks = peakfinder(counts, thresh=10, distance_=20, plot=False)

# Print the detected peaks to debug
print(f"Detected peaks: {len(peaks)}")

# Fitting Gaussian to each detected peak
params = []
for peak in peaks:
    print(f"Fitting peak at {bin_centers[peak]}")  # Debugging print statement
    
    # Define the window around the peak for fitting
    window = (bin_centers > bin_centers[peak] - 10) & (bin_centers < bin_centers[peak] + 10)
    
    # Ensure there are enough points for fitting
    if np.sum(window) < 2:  # Skip if there are too few data points
        print(f"Not enough data points to fit for peak at {bin_centers[peak]}")
        continue

    # Provide initial guess for the fitting parameters [A, mu, sigma, d]
    p0 = [counts[peak], bin_centers[peak], 1, np.min(counts)]  # Amplitude, mean, sigma, and offset (d)
    
    # Fit the Gaussian function to the data in the selected window
    try:
        print(f"Initial guess: {p0}")  # Debugging print statement
        popt, _ = curve_fit(gaussian2, bin_centers[window], counts[window], p0=p0)
        params.append(popt)  # Parameters of the fit: [Amplitude, Mu, Sigma, Offset]
        print(f"Fitted parameters: {popt}")  # Debugging print statement
    except Exception as e:
        print(f"Error fitting peak at {bin_centers[peak]}: {e}")
        continue  # Skip fitting if an error occurs

# If there are any fitting results, print and plot them
if params:
    # Visualize the results of the fit
    plt.plot(bin_centers, counts, label="Data")
    for popt in params:
        plt.plot(bin_centers, gaussian2(bin_centers, *popt), label=f"Fit Peak at {popt[1]:.2f}")
    plt.legend()
    plt.show()

    # Normalization of the peaks
    for popt in params:
        peak_center = popt[1]
        peak_amplitude = popt[0]
        print(f"Peak at {peak_center:.2f} with amplitude {peak_amplitude:.2f}")
else:
    print("No peaks were fitted successfully.")
