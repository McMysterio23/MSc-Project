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
    """
    Returns the value of a Gaussian function with an added baseline offset.

    Parameters:
    - x : array-like
        Input values where the Gaussian is evaluated.
    - A : float
        Amplitude of the Gaussian peak.
    - b : float
        Constant baseline offset.
    - fwhm : float
        Full width at half maximum of the Gaussian.
    - d : float
        Center position (mean) of the Gaussian peak.

    The function computes the Gaussian using the relationship between FWHM and 
    standard deviation: sigma = FWHM / (2 * sqrt(2 * ln(2)))
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return A * np.exp(-0.5 * ((x - d) / sigma) ** 2) + b

# Optional skewed gaussian if needed
def skewed_gaussian(x, a, loc, scale, amplitude):
    return amplitude * skewnorm.pdf(x, a, loc, scale)



def lorentzian(x, A, b, fwhm, d):
    gamma = fwhm / 2
    return A * (gamma**2 / ((x - d)**2 + gamma**2)) + b

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




def Do_Gauss_Fit_v2(bins, counts, PrintParams=False, View=False):
    """
    This function does a Gaussian fit on histogram data.

    If bins has length = len(counts), it is treated as x-data.
    If bins has length = len(counts) + 1, it is treated as bin edges.
    """

    # Determine x-data
    if len(bins) == len(counts):
        xdata = bins
    elif len(bins) == len(counts) + 1:
        xdata = (bins[:-1] + bins[1:]) / 2
    else:
        raise ValueError("Length mismatch: len(bins) must be len(counts) or len(counts) + 1")

    # Initial guess
    A0 = counts.max() - counts.min()
    b0 = counts.min()
    fwhm0 = (xdata[-1] - xdata[0]) / 10
    d0 = xdata[np.argmax(counts)]
    guess = [A0, b0, abs(fwhm0), d0]

    # Bounds
    lower_bounds = [0, 0, 1e-3, xdata[0]]
    upper_bounds = [np.inf, np.inf, xdata[-1] - xdata[0], xdata[-1]]

    try:
        popt, pcov = curve_fit(
            gaussian, xdata, counts, p0=guess,
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

    if View:
        plt.plot(xdata, counts, label='Data')
        plt.plot(xdata, gaussian(xdata, *popt), '--', label='Gaussian Fit')

        eqn_str = (
            f"$f(x) = A e^{{-0.5((x - d)/\\sigma)^2}} + b$\n"
            f"$A$ = {A:.2f} ± {A_err:.2f}, $b$ = {b:.2f} ± {b_err:.2f}\n"
            f"$\\mathrm{{FWHM}}$ = {fwhm:.2f} ± {fwhm_err:.2f} ps, $d$ = {d:.2f} ± {d_err:.2f} ps"
        )

        plt.xlabel("Δt [ps]")
        plt.ylabel("Counts")
        plt.title("Gaussian Fit")
        plt.legend(title=eqn_str, loc='upper right', fontsize='small')
        plt.tight_layout()
        plt.xlim([4600, 5100])
        plt.show()

    print("Fitted parameters (±1σ):")
    print(f"Amplitude A     : {A:.2f} ± {A_err:.2f}")
    print(f"Offset b        : {b:.2f} ± {b_err:.2f}")
    print(f"FWHM            : {fwhm:.2f} ± {fwhm_err:.2f} ps")
    print(f"Center d        : {d:.2f} ± {d_err:.2f} ps")

    if PrintParams:
        Names = ['A', 'b', 'FWHM', 'd']
        df = pd.DataFrame({
            'Parameter': Names,
            'Value': popt,
            'Uncertainty': perr
        })
        for i in range(4):
            print(f"The value of {df.loc[i, 'Parameter']} as computed from the fit appears to be {df.loc[i, 'Value']:.2f} ± {df.loc[i, 'Uncertainty']:.2f}")
        return df
    
    
    

def Do_Gauss_Fit_v3(bins, counts, PrintParams=False, View=False):
    """
    This function does a Gaussian fit on histogram data.

    If bins has length = len(counts), it is treated as x-data.
    If bins has length = len(counts) + 1, it is treated as bin edges.
    """

    # Determine x-data
    if len(bins) == len(counts):
        xdata = bins
    elif len(bins) == len(counts) + 1:
        xdata = (bins[:-1] + bins[1:]) / 2
    else:
        raise ValueError("Length mismatch: len(bins) must be len(counts) or len(counts) + 1")

    # Initial guess
    A0 = counts.max() - counts.min()
    b0 = counts.min()
    fwhm0 = (xdata[-1] - xdata[0]) / 10
    d0 = xdata[np.argmax(counts)]
    guess = [A0, b0, abs(fwhm0), d0]

    # Bounds
    lower_bounds = [0, 0, 1e-3, xdata[0]]
    upper_bounds = [np.inf, np.inf, xdata[-1] - xdata[0], xdata[-1]]

    try:
        popt, pcov = curve_fit(
            gaussian, xdata, counts, p0=guess,
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

    # Compute chi-squared
    fit_values = gaussian(xdata, *popt)
    errors = np.sqrt(counts)
    errors[errors == 0] = 1  # avoid division by zero
    chi_squared = np.sum(((counts - fit_values) / errors) ** 2)
    dof = len(counts) - len(popt)
    chi_squared_red = chi_squared / dof

    if View:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})

        # Main fit plot
        axs[0].plot(xdata, counts, 'o', label='Data')
        axs[0].plot(xdata, fit_values, '--', label='Gaussian Fit')

        eqn_str = (
            f"$f(x) = A e^{{-0.5((x - d)/\\sigma)^2}} + b$\n"
            f"$A$ = {A:.2f} ± {A_err:.2f}, $b$ = {b:.2f} ± {b_err:.2f}\n"
            f"$\\mathrm{{FWHM}}$ = {fwhm:.2f} ± {fwhm_err:.2f} ps, "
            f"$d$ = {d:.2f} ± {d_err:.2f} ps\n"
            f"$\\chi^2$ = {chi_squared:.2f}, "
            f"$\\chi^2_{{red}}$ = {chi_squared_red:.2f}"
        )

        axs[0].set_ylabel("Counts")
        axs[0].set_title("Gaussian Fit")
        axs[0].legend(title=eqn_str, loc='upper right', fontsize='small')
        axs[0].set_xlim([4600, 5100])

        # Residuals plot
        residuals = (counts - fit_values) / errors
        axs[1].axhline(0, color='gray', lw=1)
        axs[1].scatter(xdata, residuals, s=10, color='red')
        axs[1].set_xlabel("Δt [ps]")
        axs[1].set_ylabel("Residuals\n(norm.)")
        axs[1].set_xlim([4600, 5100])

        plt.tight_layout()
        plt.show()

    print("Fitted parameters (±1σ):")
    print(f"Amplitude A     : {A:.2f} ± {A_err:.2f}")
    print(f"Offset b        : {b:.2f} ± {b_err:.2f}")
    print(f"FWHM            : {fwhm:.2f} ± {fwhm_err:.2f} ps")
    print(f"Center d        : {d:.2f} ± {d_err:.2f} ps")
    print(f"Chi^2          : {chi_squared:.2f}")
    print(f"Reduced Chi^2  : {chi_squared_red:.2f}")

    if PrintParams:
        Names = ['A', 'b', 'FWHM', 'd', 'Chi^2', 'Reduced Chi^2']
        Values = list(popt) + [chi_squared, chi_squared_red]
        Uncertainties = list(perr) + [np.nan, np.nan]

        df = pd.DataFrame({
            'Parameter': Names,
            'Value': Values,
            'Uncertainty': Uncertainties
        })

        for i in range(len(df)):
            unc_str = f" ± {df.loc[i, 'Uncertainty']:.2f}" if not pd.isna(df.loc[i, 'Uncertainty']) else ""
            print(f"The value of {df.loc[i, 'Parameter']} is {df.loc[i, 'Value']:.2f}{unc_str}")

        return df
    
    
    
def Do_Gauss_Fit_v4(bins, counts, errors, PrintParams=False, View=False):
    """
    This function does a Gaussian fit on histogram data, using provided errors.
    
    Parameters:
        bins: array-like
        counts: array-like
        errors: array-like → errors on counts
        PrintParams: bool
        View: bool
    """

    if len(bins) == len(counts):
        xdata = bins
    elif len(bins) == len(counts) + 1:
        xdata = (bins[:-1] + bins[1:]) / 2
    else:
        raise ValueError("Length mismatch: len(bins) must be len(counts) or len(counts) + 1")

    # Initial guess
    A0 = counts.max() - counts.min()
    b0 = counts.min()
    fwhm0 = (xdata[-1] - xdata[0]) / 10
    d0 = xdata[np.argmax(counts)]
    guess = [A0, b0, abs(fwhm0), d0]

    # Bounds
    lower_bounds = [0, 0, 1e-3, xdata[0]]
    upper_bounds = [np.inf, np.inf, xdata[-1] - xdata[0], xdata[-1]]

    try:
        popt, pcov = curve_fit(
            gaussian, xdata, counts, p0=guess,
            bounds=(lower_bounds, upper_bounds),
            sigma=errors,
            absolute_sigma=True
        )
    except RuntimeError as e:
        print("Fit failed:", e)
        return

    A, b, fwhm, d = popt
    perr = np.sqrt(np.diag(pcov))
    A_err, b_err, fwhm_err, d_err = perr

    # Compute chi-squared
    fit_values = gaussian(xdata, *popt)
    chi_squared = np.sum(((counts - fit_values) / errors) ** 2)
    dof = len(counts) - len(popt)
    chi_squared_red = chi_squared / dof

    if View:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})

        axs[0].plot(xdata, counts, 'o', label='Data')
        axs[0].plot(xdata, fit_values, '--', label='Gaussian Fit')

        eqn_str = (
            f"$f(x) = A e^{{-0.5((x - d)/\\sigma)^2}} + b$\n"
            f"$A$ = {A:.2f} ± {A_err:.2f}, $b$ = {b:.2f} ± {b_err:.2f}\n"
            f"$\\mathrm{{FWHM}}$ = {fwhm:.2f} ± {fwhm_err:.2f} ps, "
            f"$d$ = {d:.2f} ± {d_err:.2f} ps\n"
            f"$\\chi^2$ = {chi_squared:.2f}, "
            f"$\\chi^2_{{red}}$ = {chi_squared_red:.2f}"
        )

        axs[0].set_ylabel("Counts")
        axs[0].set_title("Gaussian Fit")
        axs[0].legend(title=eqn_str, loc='upper right', fontsize='small')
        axs[0].set_xlim([4600, 5100])

        residuals = (counts - fit_values) / errors
        axs[1].axhline(0, color='gray', lw=1)
        axs[1].scatter(xdata, residuals, s=10, color='red')
        axs[1].set_xlabel("Δt [ps]")
        axs[1].set_ylabel("Residuals\n(norm.)")
        axs[1].set_xlim([4600, 5100])

        plt.tight_layout()
        plt.show()

    print("Fitted parameters (±1σ):")
    print(f"Amplitude A     : {A:.2f} ± {A_err:.2f}")
    print(f"Offset b        : {b:.2f} ± {b_err:.2f}")
    print(f"FWHM            : {fwhm:.2f} ± {fwhm_err:.2f} ps")
    print(f"Center d        : {d:.2f} ± {d_err:.2f} ps")
    print(f"Chi^2          : {chi_squared:.2f}")
    print(f"Reduced Chi^2  : {chi_squared_red:.2f}")

    if PrintParams:
        Names = ['A', 'b', 'FWHM', 'd', 'Chi^2', 'Reduced Chi^2']
        Values = list(popt) + [chi_squared, chi_squared_red]
        Uncertainties = list(perr) + [np.nan, np.nan]

        df = pd.DataFrame({
            'Parameter': Names,
            'Value': Values,
            'Uncertainty': Uncertainties
        })

        for i in range(len(df)):
            unc_str = f" ± {df.loc[i, 'Uncertainty']:.2f}" if not pd.isna(df.loc[i, 'Uncertainty']) else ""
            print(f"The value of {df.loc[i, 'Parameter']} is {df.loc[i, 'Value']:.2f}{unc_str}")

        return df


def Do_Lorentzian_Fit(bins, counts, errors, PrintParams=False, View=False, DebugPrints=False):
    """
    This function performs a Lorentzian fit on histogram data using provided errors.
    
    Parameters:
        bins: array-like
        counts: array-like
        errors: array-like → errors on counts
        PrintParams: bool → whether to print each parameter value with uncertainty
        View: bool → whether to plot data, fit, and residuals
    """

    if len(bins) == len(counts):
        xdata = bins
    elif len(bins) == len(counts) + 1:
        xdata = (bins[:-1] + bins[1:]) / 2
    else:
        raise ValueError("Length mismatch: len(bins) must be len(counts) or len(counts) + 1")

    # Initial guess
    A0 = counts.max()  # Better than max - min
    b0 = counts.min()
    fwhm0 = (xdata[-1] - xdata[0]) / 10
    d0 = xdata[np.argmax(counts)]
    guess = [A0, b0, abs(fwhm0), d0]

    # Bounds
    lower_bounds = [0, 0, 1e-3, xdata[0]]
    upper_bounds = [np.inf, np.inf, xdata[-1] - xdata[0], xdata[-1]]

    try:
        popt, pcov = curve_fit(
            lorentzian, xdata, counts, p0=guess,
            bounds=(lower_bounds, upper_bounds),
            sigma=errors,
            absolute_sigma=True
        )
    except RuntimeError as e:
        print("Fit failed:", e)
        return

    A, b, fwhm, d = popt
    perr = np.sqrt(np.diag(pcov))
    A_err, b_err, fwhm_err, d_err = perr

    # Compute chi-squared
    fit_values = lorentzian(xdata, *popt)
    chi_squared = np.sum(((counts - fit_values) / errors) ** 2)
    dof = len(counts) - len(popt)
    chi_squared_red = chi_squared / dof

    if View:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})

        axs[0].plot(xdata, counts, 'o', label='Data')
        axs[0].plot(xdata, fit_values, '--', label='Lorentzian Fit')

        eqn_str = (
            f"$f(x) = A \\cdot \\frac{{\\gamma^2}}{{(x - d)^2 + \\gamma^2}} + b$\n"
            f"$A$ = {A:.2f} ± {A_err:.2f}, $b$ = {b:.2f} ± {b_err:.2f}\n"
            f"$\\mathrm{{FWHM}}$ = {fwhm:.2f} ± {fwhm_err:.2f} ps, "
            f"$d$ = {d:.2f} ± {d_err:.2f} ps\n"
            f"$\\chi^2$ = {chi_squared:.2f}, "
            f"$\\chi^2_{{red}}$ = {chi_squared_red:.2f}"
        )

        axs[0].set_ylabel("Counts")
        axs[0].set_title("Lorentzian Fit")
        axs[0].legend(title=eqn_str, loc='upper right', fontsize='small')
        axs[0].set_xlim([4600, 5100])

        residuals = (counts - fit_values) / errors
        axs[1].axhline(0, color='gray', lw=1)
        axs[1].scatter(xdata, residuals, s=10, color='red')
        axs[1].set_xlabel("Δt [ps]")
        axs[1].set_ylabel("Residuals\n(norm.)")
        axs[1].set_xlim([4600, 5100])

        plt.tight_layout()
        plt.show()
        
    if DebugPrints :
        print("Fitted parameters (±1σ):")
        print(f"Amplitude A     : {A:.2f} ± {A_err:.2f}")
        print(f"Offset b        : {b:.2f} ± {b_err:.2f}")
        print(f"FWHM            : {fwhm:.2f} ± {fwhm_err:.2f} ps")
        print(f"Center d        : {d:.2f} ± {d_err:.2f} ps")
        print(f"Chi^2          : {chi_squared:.2f}")
        print(f"Reduced Chi^2  : {chi_squared_red:.2f}")

    if PrintParams:
        Names = ['A', 'b', 'FWHM', 'd', 'Chi^2', 'Reduced Chi^2']
        Values = list(popt) + [chi_squared, chi_squared_red]
        Uncertainties = list(perr) + [np.nan, np.nan]

        df = pd.DataFrame({
            'Parameter': Names,
            'Value': Values,
            'Uncertainty': Uncertainties
        })

        # for i in range(len(df)):
        #     unc_str = f" ± {df.loc[i, 'Uncertainty']:.2f}" if not pd.isna(df.loc[i, 'Uncertainty']) else ""
        #     print(f"The value of {df.loc[i, 'Parameter']} is {df.loc[i, 'Value']:.2f}{unc_str}")

        return df



def Do_Gauss_Fit_v5(bins, counts, errors = None, PrintParams=False, View=False, DebugPrints=False):
    """
    Perform Gaussian fit on histogram data, trying two initial guesses:
    one with a background and one assuming no background. The better fit
    (based on reduced chi-squared) is retained.

    Parameters:
        bins: array-like - bin edges or centers
        counts: array-like - histogram counts
        errors: array-like - uncertainty on counts
        PrintParams: bool - whether to print fit results
        View: bool - whether to plot fit and residuals
        DebugPrints: bool - whether to print debug outputs during fitting
    """
    
    if errors is None:
        # Avoid sqrt of negative or zero counts
        errors = np.sqrt(np.where(counts > 0, counts, 1))

    if len(bins) == len(counts):
        xdata = bins
    elif len(bins) == len(counts) + 1:
        xdata = (bins[:-1] + bins[1:]) / 2
    else:
        raise ValueError("Length mismatch: len(bins) must be len(counts) or len(counts) + 1")

    def run_fit(guess):
        try:
            popt, pcov = curve_fit(
                gaussian, xdata, counts, p0=guess,
                bounds=(lower_bounds, upper_bounds),
                sigma=errors, absolute_sigma=True
            )
            fit_values = gaussian(xdata, *popt)
            chi2 = np.sum(((counts - fit_values) / errors) ** 2)
            dof = len(counts) - len(popt)
            chi2_red = chi2 / dof
            return popt, pcov, fit_values, chi2, chi2_red
        except RuntimeError as e:
            if DebugPrints:
                print("Fit failed:", e)
            return None, None, None, np.inf, np.inf

    # --- Bounds ---
    lower_bounds = [0, 0, 1e-3, xdata[0]]
    upper_bounds = [np.inf, np.inf, xdata[-1] - xdata[0], xdata[-1]]

    # --- First guess (with background) ---
    A0 = counts.max() - counts.min()
    b0 = counts.min()
    fwhm0 = (xdata[-1] - xdata[0]) / 10
    d0 = xdata[np.argmax(counts)]
    guess1 = [A0, b0, abs(fwhm0), d0]

    # --- Second guess (no background) ---
    guess2 = [counts.max(), 0, abs(fwhm0), d0]

    # --- Run both fits ---
    popt1, pcov1, fit1, chi2_1, chi2_red1 = run_fit(guess1)
    popt2, pcov2, fit2, chi2_2, chi2_red2 = run_fit(guess2)

    # --- Choose best fit ---
    if chi2_red1 <= chi2_red2:
        popt, pcov, fit_values, chi2, chi2_red = popt1, pcov1, fit1, chi2_1, chi2_red1
    else:
        popt, pcov, fit_values, chi2, chi2_red = popt2, pcov2, fit2, chi2_2, chi2_red2

    A, b, fwhm, d = popt
    perr = np.sqrt(np.diag(pcov))
    A_err, b_err, fwhm_err, d_err = perr

    if View:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})

        axs[0].plot(xdata, counts, 'o', label='Data')
        axs[0].plot(xdata, fit_values, '--', label='Gaussian Fit')

        eqn_str = (
            f"$f(x) = A e^{{-0.5((x - d)/\\sigma)^2}} + b$\n"
            f"$A$ = {A:.2f} ± {A_err:.2f}, $b$ = {b:.2f} ± {b_err:.2f}\n"
            f"$\\mathrm{{FWHM}}$ = {fwhm:.2f} ± {fwhm_err:.2f} ps, "
            f"$d$ = {d:.2f} ± {d_err:.2f} ps\n"
            f"$\\chi^2$ = {chi2:.2f}, "
            f"$\\chi^2_{{red}}$ = {chi2_red:.2f}"
        )

        axs[0].set_ylabel("Counts")
        axs[0].set_title("Gaussian Fit")
        axs[0].legend(title=eqn_str, loc='upper right', fontsize='small')

        residuals = (counts - fit_values) / errors
        axs[1].axhline(0, color='gray', lw=1)
        axs[1].scatter(xdata, residuals, s=10, color='red')
        axs[1].set_xlabel("Δt [ps]")
        axs[1].set_ylabel("Residuals\n(norm.)")

        plt.tight_layout()
        plt.show()

    if DebugPrints:
        print("Fitted parameters (±1σ):")
        print(f"Amplitude A     : {A:.2f} ± {A_err:.2f}")
        print(f"Offset b        : {b:.2f} ± {b_err:.2f}")
        print(f"FWHM            : {fwhm:.2f} ± {fwhm_err:.2f} ps")
        print(f"Center d        : {d:.2f} ± {d_err:.2f} ps")
        print(f"Chi^2           : {chi2:.2f}")
        print(f"Reduced Chi^2   : {chi2_red:.2f}")

    if PrintParams:
        Names = ['A', 'b', 'FWHM', 'd', 'Chi^2', 'Reduced Chi^2']
        Values = list(popt) + [chi2, chi2_red]
        Uncertainties = list(perr) + [np.nan, np.nan]

        df = pd.DataFrame({
            'Parameter': Names,
            'Value': Values,
            'Uncertainty': Uncertainties
        })

        # for i in range(len(df)):
        #     unc_str = f" ± {df.loc[i, 'Uncertainty']:.2f}" if not pd.isna(df.loc[i, 'Uncertainty']) else ""
        #     print(f"The value of {df.loc[i, 'Parameter']} is {df.loc[i, 'Value']:.2f}{unc_str}")

        return df
    
    
    
    
    
def sech2(x, A, b, tau, t0):
    
    """
    Computes a scaled and shifted squared hyperbolic secant function.

    Parameters:
        x (float or np.ndarray): Input value(s).
        A (float): Amplitude of the peak.
        b (float): Baseline offset.
        tau (float): Controls the width of the peak.
        t0 (float): Center of the peak.

    Returns:
        float or np.ndarray: The value of the sech² function at x.
    """
    
    return A * (1 / np.cosh((x - t0) / tau))**2 + b

def Do_Sech2_Fit(bins, counts, errors = None, PrintParams=False, View=False, DebugPrints=False):
    """
    Fit histogram data using a sech^2 pulse shape, common for ultrafast laser pulses.

    Parameters:
        bins: array-like - bin edges or centers
        counts: array-like - histogram counts
        errors: array-like - uncertainty on counts
        PrintParams: bool - whether to print fit results
        View: bool - whether to plot fit and residuals
        DebugPrints: bool - verbose output for debugging
    """

    if errors is None:
        # Avoid sqrt of negative or zero counts
        errors = np.sqrt(np.where(counts > 0, counts, 1))
    
    if len(bins) == len(counts):
        xdata = bins
    elif len(bins) == len(counts) + 1:
        xdata = (bins[:-1] + bins[1:]) / 2
    else:
        raise ValueError("Length mismatch: len(bins) must be len(counts) or len(counts) + 1")

    # Initial parameter guess
    A0 = counts.max() - counts.min()
    b0 = counts.min()
    tau0 = (xdata[-1] - xdata[0]) / 10
    t0 = xdata[np.argmax(counts)]
    guess = [A0, b0, abs(tau0), t0]

    # Bounds for fitting
    lower_bounds = [0, 0, 1e-3, xdata[0]]
    upper_bounds = [np.inf, np.inf, xdata[-1] - xdata[0], xdata[-1]]

    try:
        popt, pcov = curve_fit(
            sech2, xdata, counts, p0=guess,
            bounds=(lower_bounds, upper_bounds),
            sigma=errors, absolute_sigma=True
        )
    except RuntimeError as e:
        print("Fit failed:", e)
        return

    A, b, tau, t0 = popt
    perr = np.sqrt(np.diag(pcov))
    A_err, b_err, tau_err, t0_err = perr

    # Compute chi-squared
    fit_values = sech2(xdata, *popt)
    chi2 = np.sum(((counts - fit_values) / errors)**2)
    dof = len(counts) - len(popt)
    chi2_red = chi2 / dof
    
    fwhm = list(popt)[2] * 1.76
    efwhm = list(perr)[2] * 1.76
    

    if View:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})

        axs[0].plot(xdata, counts, 'o', label='Data')
        axs[0].plot(xdata, fit_values, '--', label='sech² Fit')

        eqn_str = (
            r"$f(t) = A \cdot \mathrm{sech}^2\left(\frac{t - t_0}{\tau}\right) + b$" "\n"
            f"$A$ = {A:.2f} ± {A_err:.2f}, $b$ = {b:.2f} ± {b_err:.2f}\n"
            f"$\\tau$ = {tau:.2f} ± {tau_err:.2f} ps, $t_0$ = {t0:.2f} ± {t0_err:.2f} ps\n"
            f"$\\chi^2$ = {chi2:.2f}, $\\chi^2_{{red}}$ = {chi2_red:.2f}"
        )

        axs[0].set_ylabel("Counts")
        axs[0].set_title("Sech² Pulse Fit")
        axs[0].legend(title=eqn_str, loc='upper right', fontsize='small')

        residuals = (counts - fit_values) / errors
        axs[1].axhline(0, color='gray', lw=1)
        axs[1].scatter(xdata, residuals, s=10, color='red')
        axs[1].set_xlabel("Δt [ps]")
        axs[1].set_ylabel("Residuals\n(norm.)")

        plt.tight_layout()
        plt.show()

    if DebugPrints :
        print("Fitted parameters :")
        print(f"Amplitude A     : {A:.2f} ± {A_err:.2f}")
        print(f"Offset b        : {b:.2f} ± {b_err:.2f}")
        print(f"τ (width param) : {tau:.2f} ± {tau_err:.2f} ps")
        print(f"Center t₀       : {t0:.2f} ± {t0_err:.2f} ps")
        print(f'FWHM            : {fwhm:.2f} ± {efwhm:.2f} ps')
        print(f"Chi²            : {chi2:.2f}")
        print(f"Reduced Chi²    : {chi2_red:.2f}")

    if PrintParams:
        Names = ['A', 'b', 'τ', 't₀', 'FWHM', 'Chi²', 'Reduced Chi²']
        Values = list(popt) + [fwhm, chi2, chi2_red]
        Uncertainties = list(perr) + [efwhm, np.nan, np.nan]

        df = pd.DataFrame({
            'Parameter': Names,
            'Value': Values,
            'Uncertainty': Uncertainties
        })


        return df