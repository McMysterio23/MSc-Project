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