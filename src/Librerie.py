# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 19:36:09 2025

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


def gaussian(x, A, b, fwhm, d):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return A * np.exp(-0.5 * ((x - d) / sigma) ** 2) + b

# Optional skewed gaussian if needed
def skewed_gaussian(x, a, loc, scale, amplitude):
    return amplitude * skewnorm.pdf(x, a, loc, scale)

def Do_Gauss_Fit(bins, counts, PrintParams = False, View = False):
    
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

    
    if(View):
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
        plt.xlim([4600,5100])
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
            print(f"The value of {df.loc[i, 'Parameter']} as computed from the fit appears to be {df.loc[i, 'Value']:.2f} ± {df.loc[i, 'Uncertainty']:.2f}")
    
        return df
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def peakfinder22(arr, thresh=None, distance_=1, plot=False):
    """
    Identifies positive peaks in a 1D array based on the first derivative.

    Parameters:
    - arr: Input array to search for peaks.
    - thresh: Threshold multiplier for peak sensitivity (default: 2).
    - distance_: Minimum spacing between detected peaks.
    - plot: If True, plots intermediate steps for visualization.

    Returns:
    - Array of peak indices where the most significant peaks occur.
    """
    
    # First derivative (rate of change)
    diff_arr = np.diff(arr)
    
    # Find local maxima by checking where the first derivative changes from positive to negative
    peak_candidates = np.argwhere((diff_arr[:-1] > 0) & (diff_arr[1:] < 0)).flatten() + 1  # Peaks are where the first derivative changes sign
    
    if thresh is not None:
        # Apply threshold to peak candidates by considering only peaks with sufficient amplitude
        peak_candidates = [i for i in peak_candidates if arr[i] > thresh]
    
    # Handle the case where no peaks are detected
    if len(peak_candidates) == 0:
        print("No peaks detected.")
        return []

    # Apply distance filtering (minimum spacing between peaks)
    peak_idx = decluster_indices(arr, peak_candidates, distance=distance_)

    # Plotting (optional)
    if plot:
        plt.plot(arr/arr.max())
        plt.plot(peak_idx, arr[peak_idx]/arr.max(), 'ro')  # Red circles for detected peaks
        plt.title("Detected Peaks")
        plt.show()

    return peak_idx

def decluster_indices(array, idxs, distance=1):
    """
    Removes indices that are too close to each other, keeping only the most prominent peaks.

    Parameters:
    - array: The data array from which peaks are identified.
    - idxs: List or array of peak indices to decluster.
    - distance: The minimum distance (in terms of indices) that peaks should be apart.

    Returns:
    - Declustered list of peak indices.
    """
    if len(idxs) == 0:
        return []

    declustered = [idxs[0]]  # Keep the first peak

    for idx in idxs[1:]:
        # Compare the current peak to the last kept peak
        if abs(idx - declustered[-1]) >= distance:
            declustered.append(idx)  # Add it if it's far enough from the last peak

    return np.array(declustered)
