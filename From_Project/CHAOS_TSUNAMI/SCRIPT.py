# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 16:16:17 2025

@author: Maccarinelli
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

# --- Dynamically find path of the current script and look for .csv ---
current_dir = Path(__file__).resolve().parent
csv_files = list(current_dir.glob("*.csv"))

if len(csv_files) != 1:
    raise RuntimeError(f"Expected exactly one CSV file in folder, found {len(csv_files)}.")

csv_file = csv_files[0]

# --- Load data ---
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

peak_index1 = np.argmax(hist1)
peak_index2 = np.argmax(hist2)
peak_index3 = np.argmax(hist3)
peak_index4 = np.argmax(hist4)

peak_position1 = positions[peak_index1]
peak_position2 = positions[peak_index2]
peak_position3 = positions[peak_index3]
peak_position4 = positions[peak_index4]

positions1 = positions.copy() - peak_position1
positions2 = positions.copy() - peak_position2
positions3 = positions.copy() - peak_position3
positions4 = positions.copy() - peak_position4


selection_mask = (positions > -12000) & (positions < 12000)
#Restricting the arrays to the current interval being studied !
positions1 = positions1[selection_mask]
positions2 = positions2[selection_mask]
positions3 = positions3[selection_mask]
positions4 = positions4[selection_mask]

hist1 = hist1[selection_mask]
hist2 = hist2[selection_mask]
hist3 = hist3[selection_mask]
hist4 = hist4[selection_mask]

ehist1 = ehist1[selection_mask]
ehist2 = ehist2[selection_mask]
ehist3 = ehist3[selection_mask]
ehist4 = ehist4[selection_mask]


# Selected_Ncounts = Ncounts[selection_mask]
# selected_errors = ErrNcounts[selection_mask]

# # Filter out non valid errors !!! (Avoiding errors that are NaN or Zero)
# valid_mask = (selected_errors > 0) & np.isfinite(selected_errors) & np.isfinite(Selected_Ncounts)
# Selection_Taus = Selection_Taus[valid_mask]
# Selected_Ncounts = Selected_Ncounts[valid_mask]
# selected_errors = selected_errors[valid_mask]


def compute_fwhm(pos, hist):
    interp_func = interp1d(pos, hist, kind='cubic', fill_value="extrapolate")
    peak_height = np.max(hist)
    half_max = peak_height / 2.0
    x_fine = np.linspace(pos[0], pos[-1], 10000)
    y_fine = interp_func(x_fine)
    indices = np.where(np.diff(np.sign(y_fine - half_max)))[0]
    if len(indices) >= 2:
        x_left = x_fine[indices[0]]
        x_right = x_fine[indices[-1]]
        fwhm = x_right - x_left
        return x_fine, y_fine, x_left, x_right, half_max, fwhm
    else:
        return None, None, None, None, None, None

x_fine1, y_fine1, xL1, xR1, hmax1, fwhm1 = compute_fwhm(positions1, hist1)
x_fine2, y_fine2, xL2, xR2, hmax2, fwhm2 = compute_fwhm(positions2, hist2)
x_fine3, y_fine3, xL3, xR3, hmax3, fwhm3 = compute_fwhm(positions3, hist3)
# x_fine4, y_fine4, xL4, xR4, hmax4, fwhm4 = compute_fwhm(positions4, hist4)

plt.figure(figsize=(12, 8))

# Histogram 1
plt.errorbar(positions1, hist1, yerr=ehist1, fmt='*', color='black',
             capsize=2, ecolor='orange', label='TCSPC SYNC vs DET2')
plt.plot(x_fine1, y_fine1, 'k--', alpha=0.5)
# plt.axhline(hmax1, color='black', linestyle='--', alpha=0.25)
# plt.axvline(xL1, color='black', linestyle='--', alpha=0.25)
# plt.axvline(xR1, color='black', linestyle='--', alpha=0.25)
# plt.hlines(hmax1, xL1, xR1, colors='black', linewidth=2 , alpha=0.7)
           # label=f'FWHM Sync vs Det2 = {fwhm1:.2f}', alpha=0.7)
# plt.text(0, 4.65e+03, f'{fwhm1:.2f}ps', color='black', ha='center')

# Histogram 2
plt.errorbar(positions2, hist2, yerr=ehist2, fmt='.', color='brown',
             capsize=2, ecolor='orange', label='TCSPC SYNC vs DET3')
plt.plot(x_fine2, y_fine2, 'brown', linestyle='--', alpha=0.5)
# plt.axhline(hmax2, color='brown', linestyle='--', alpha=0.25)
# plt.axvline(xL2, color='brown', linestyle='--', alpha=0.25)
# plt.axvline(xR2, color='brown', linestyle='--', alpha=0.25)
# plt.hlines(hmax2, xL2, xR2, colors='brown', linewidth=2, alpha=0.7)
           # label=f'FWHM Sync vs Det3 = {fwhm2:.2f}',
# plt.text(0, 7.5e+03, f'{fwhm2:.2f}ps', color='brown', ha='center')


# Histogram 3
plt.errorbar(positions3, hist3, yerr=ehist3, fmt='.', color='green',
             capsize=2, ecolor='orange', label='HBT Detector2 vs Detector3')
plt.plot(x_fine3, y_fine3, 'green', linestyle='--', alpha=0.5)
# plt.axhline(hmax3, color='green', linestyle='--', alpha=0.25)
# plt.axvline(xL3, color='green', linestyle='--', alpha=0.25)
# plt.axvline(xR3, color='green', linestyle='--', alpha=0.25)
# plt.hlines(hmax3, xL3, xR3, colors='green', linewidth=2, alpha=0.7)
           # label='FWHM HBT Det2 vs Det3 = 46.27 ps', alpha=0.7)
# plt.text(0, 2.81e+04, '46.27 ps', color='Green', ha='center')

#  Histogram 4
plt.errorbar(positions4, hist4, yerr=ehist4, fmt='.', color='blue',
             capsize=2, ecolor='orange', label='HBT Detector3 vs Detector2')
plt.plot(positions4, hist4, 'blue', linestyle='--', alpha=0.5)
# plt.axhline(hmax4, color='blue', linestyle='--', alpha=0.25)
# plt.axvline(xL4, color='blue', linestyle='--', alpha=0.25)
# plt.axvline(xR4, color='blue', linestyle='--', alpha=0.25)
# plt.hlines(hmax4, xL4, xR4, colors='blue', linewidth=2,
           # label=f'FWHM HBT Det3 vs Det2= {fwhm4:.2f}', alpha=0.7)
# plt.text(0, 2.51e+04, f'{fwhm4:.2f}ps', color='blue', ha='center')


# #  Histogram 5
# plt.errorbar(-positions4, hist4, yerr=ehist4, fmt='.', color='purple',
#              capsize=2, ecolor='orange', label='HBT Detector3 vs Detector2')
# plt.plot(x_fine4, y_fine4, 'purple', linestyle='--', alpha=0.5)
# plt.axhline(hmax4, color='purple', linestyle='--', alpha=0.25)
# plt.axvline(xL4, color='purple', linestyle='--', alpha=0.25)
# plt.axvline(xR4, color='purple', linestyle='--', alpha=0.25)
# plt.hlines(hmax4, xL4, xR4, colors='purple', linewidth=2,
#            label=f'FWHM HBT Det3 vs Det2= {fwhm4:.2f}', alpha=0.7)
# plt.text(0, 2.51e+04, f'{fwhm4:.2f}ps', color='purple', ha='center')



# plt.axvline(174.4, color='brown', linestyle='-.', alpha=0.85, label = 'TCSPC DET3 Second Sidepeak')
# plt.axvline(169.8, color='green', linestyle='-.', alpha=0.85, label = 'TCSPC DET2 Second Sidepeak')
# plt.axvline(75.2, color='black', linestyle='-.', alpha=0.85, label = 'HBT Second Sidepeak')

# plt.xlim(-700, +700)
plt.xlim(-700, +700)

#LOG SCALE STILL DOESN'T WORK !!!!
# plt.yscale('log')
# plt.ylim(0.1, +10.975071168816319)
plt.ylabel("Counts (a.u.)", fontsize = 16)

# Customize ticks
plt.tick_params(axis='both', which='major', labelsize=16, length=8, width=2)
plt.tick_params(axis='both', which='minor', labelsize=16, length=10, width=2)

# Legend with larger font
plt.legend(fontsize=14,           # Bigger text
           handlelength=3,        # Length of the legend lines
           handleheight=2,        # Height of the legend handles (vertical space)
           handletextpad=1.5,     # Space between handle and text
           markerscale=2)
# plt.grid(True)
plt.tight_layout()
plt.show()
