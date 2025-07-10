# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 14:21:23 2025

@author: Maccarinelli
"""
"""
This Script is designed so that it can plot the first peak of a long measurement whose maximum x coordinate is longer
than the repetition rate of the Ti:Sa laser. This is now fixed so that every peak of the Start-Stop measurements
is now at the expected position.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

# --- Trova il file CSV ---
current_dir = Path(__file__).resolve().parent
csv_files = list(current_dir.glob("*.csv"))

if len(csv_files) != 1:
    raise RuntimeError(f"Expected exactly one CSV file in folder, found {len(csv_files)}.")

csv_file = csv_files[0]

# --- Caricamento dati ---
df = pd.read_csv(csv_file, sep=";")

positions = df.iloc[:, 0].values
hist1 = df.iloc[:, 1].values
hist2 = df.iloc[:, 2].values
hist3 = df.iloc[:, 3].values
hist4 = df.iloc[:, 4].values

ehist1 = np.sqrt(hist1)
ehist2 = np.sqrt(hist2)
ehist3 = np.sqrt(hist3)
ehist4 = np.sqrt(hist4)

# --- Funzione per trovare il massimo entro i primi 12500ps ---
def find_local_peak(pos, hist, window_ps=12500):
    mask = (pos >= pos[0]) & (pos <= pos[0] + window_ps)
    local_pos = pos[mask]
    local_hist = hist[mask]
    if len(local_hist) == 0:
        raise ValueError("Empty interval for local peak search.")
    peak_index = np.argmax(local_hist)
    return local_pos[peak_index]

# Trova i picchi locali nei primi 12500 ps
peak_position1 = find_local_peak(positions, hist1)
peak_position2 = find_local_peak(positions, hist2)
peak_position3 = find_local_peak(positions, hist3)
peak_position4 = find_local_peak(positions, hist4)

# Applica reshift
positions1 = positions - peak_position1
positions2 = positions - peak_position2
positions3 = positions - peak_position3
positions4 = positions - peak_position4

# # Applica selezione temporale (dopo il reshift)
# selection_mask = (positions1 > -25000) & (positions1 < 25000)

# positions1 = positions1[selection_mask]
# positions2 = positions2[selection_mask]
# positions3 = positions3[selection_mask]
# positions4 = positions4[selection_mask]

# hist1 = hist1[selection_mask]
# hist2 = hist2[selection_mask]
# hist3 = hist3[selection_mask]
# hist4 = hist4[selection_mask]

# ehist1 = ehist1[selection_mask]
# ehist2 = ehist2[selection_mask]
# ehist3 = ehist3[selection_mask]
# ehist4 = ehist4[selection_mask]

# --- Funzione FWHM ---
# def compute_fwhm(pos, hist):
#     interp_func = interp1d(pos, hist, kind='cubic', fill_value="extrapolate")
#     peak_height = np.max(hist)
#     half_max = peak_height / 2.0
#     x_fine = np.linspace(pos[0], pos[-1], 10000)
#     y_fine = interp_func(x_fine)
#     indices = np.where(np.diff(np.sign(y_fine - half_max)))[0]
#     if len(indices) >= 2:
#         x_left = x_fine[indices[0]]
#         x_right = x_fine[indices[-1]]
#         fwhm = x_right - x_left
#         return x_fine, y_fine, x_left, x_right, half_max, fwhm
#     else:
#         return None, None, None, None, None, None


def compute_fwhm(pos, hist, window_width_ps=3000):
    """
    Computes the FWHM of the main peak, restricting the analysis to a local window
    around the maximum to avoid capturing side peaks.
    
    Parameters:
        pos (np.array): x-axis positions (e.g., ps)
        hist (np.array): histogram counts
        window_width_ps (int): half-width of the window around the peak [default: 3000 ps]

    Returns:
        x_fine, y_fine: interpolated data for plotting
        x_left, x_right: positions at FWHM
        half_max: value of half max
        fwhm: computed full width at half maximum
    """
    # Find index of the maximum
    peak_idx = np.argmax(hist)
    peak_pos = pos[peak_idx]

    # Define window around peak
    window_mask = (pos >= peak_pos - window_width_ps) & (pos <= peak_pos + window_width_ps)
    pos_window = pos[window_mask]
    hist_window = hist[window_mask]

    # Interpolation
    interp_func = interp1d(pos_window, hist_window, kind='cubic', fill_value="extrapolate")
    x_fine = np.linspace(pos_window[0], pos_window[-1], 10000)
    y_fine = interp_func(x_fine)

    # Compute FWHM
    peak_height = np.max(hist_window)
    half_max = peak_height / 2.0
    indices = np.where(np.diff(np.sign(y_fine - half_max)))[0]

    if len(indices) >= 2:
        x_left = x_fine[indices[0]]
        x_right = x_fine[indices[-1]]
        fwhm = x_right - x_left
        return x_fine, y_fine, x_left, x_right, half_max, fwhm
    else:
        return x_fine, y_fine, None, None, half_max, None

# Calcolo FWHM
x_fine1, y_fine1, xL1, xR1, hmax1, fwhm1 = compute_fwhm(positions1, hist1)
x_fine2, y_fine2, xL2, xR2, hmax2, fwhm2 = compute_fwhm(positions2, hist2)
x_fine3, y_fine3, xL3, xR3, hmax3, fwhm3 = compute_fwhm(positions3, hist3)
x_fine4, y_fine4, xL4, xR4, hmax4, fwhm4 = compute_fwhm(positions4, hist4)

# --- Plot ---
plt.figure(figsize=(18, 8))

plt.errorbar(positions1 /1000, hist1, yerr=ehist1, fmt='*', color='black',
             capsize=2, ecolor='orange', label='TCSPC SYNC vs DET2')
plt.plot(x_fine1/1000, y_fine1, 'k--', alpha=0.5)

plt.errorbar(positions2/1000, hist2, yerr=ehist2, fmt='.', color='brown',
             capsize=2, ecolor='orange', label='TCSPC SYNC vs DET3')
plt.plot(x_fine2/1000, y_fine2, 'brown', linestyle='--', alpha=0.5)

plt.errorbar(positions3/1000, hist3, yerr=ehist3, fmt='.', color='green',
             capsize=2, ecolor='orange', label='HBT Detector2 vs Detector3')
plt.plot(x_fine3/1000, y_fine3, 'green', linestyle='--', alpha=0.5)

plt.errorbar(positions4/1000, hist4, yerr=ehist4, fmt='.', color='blue',
             capsize=2, ecolor='orange', label='HBT Detector3 vs Detector2')
plt.plot(positions4/1000, hist4, 'blue', linestyle='--', alpha=0.5)

plt.xlabel("Time Delay (ns)")
plt.ylabel("Counts (a.u.)")
plt.xlim(-2, 39)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 8))

plt.errorbar(positions1  , hist1, yerr=ehist1, fmt='*', color='black',
             capsize=2, ecolor='orange', label='TCSPC SYNC vs DET2')
plt.plot(x_fine1 , y_fine1, 'k--', alpha=0.5)

plt.errorbar(positions2 , hist2, yerr=ehist2, fmt='.', color='brown',
             capsize=2, ecolor='orange', label='TCSPC SYNC vs DET3')
plt.plot(x_fine2 , y_fine2, 'brown', linestyle='--', alpha=0.5)

plt.errorbar(positions3 , hist3, yerr=ehist3, fmt='.', color='green',
             capsize=2, ecolor='orange', label='HBT Detector2 vs Detector3')
plt.plot(x_fine3 , y_fine3, 'green', linestyle='--', alpha=0.5)

plt.errorbar(positions4 , hist4, yerr=ehist4, fmt='.', color='blue',
             capsize=2, ecolor='orange', label='HBT Detector3 vs Detector2')
plt.plot(positions4 , hist4, 'blue', linestyle='--', alpha=0.5)

plt.xlabel("Time Delay (ps)")
plt.ylabel("Counts (a.u.)")
plt.yscale('log')
plt.ylim(0.2)
plt.xlim(-800, +800)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
