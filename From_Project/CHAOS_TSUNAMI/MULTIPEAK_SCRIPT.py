# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 10:32:26 2025

@author: Maccarinelli


Not Currently working with the final set of measurements !!!
"""



"""
This Script is designed to reach an approximated version of the function that lies under the HBT plot from the TiSa (TSUNAMI) start stop measurements
Keep in mind that this file has to be executed in the same folder of the csv file containing the histograms !!!
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path
from src.Librerie import Do_Gauss_Fit_v4, gaussian, sech2, lorentzian
from scipy.special import voigt_profile
from scipy.optimize import curve_fit

def make_wrapped_model(fixed_params):
    param_names = ["A1", "FWHM_1", "A2", "FWHM_2", "C2", "A3", "FWHM_3", "C3", "A4", "FWHM_4", "C4",
                   "A5", "FWHM_5", "C5",
                   "A6", "FWHM_6", "C6",
                   "A7", "FWHM_7", "C7",
                   "A8", "FWHM_8", "C8"]

    def wrapped_model(x, *free_values):
        values = {}
        i = 0
        for name in param_names:
            if name in fixed_params:
                values[name] = fixed_params[name]
            else:
                values[name] = free_values[i]
                i += 1

        return (
            sech2(x, values["A1"], 0, values["FWHM_1"]/1.76, 0) +
            sech2(x, values["A2"], 0, values["FWHM_2"]/1.76, values["C2"])+
            gaussian(x, values["A3"], 0, values["FWHM_3"], values["C3"])+
            gaussian(x, values["A4"], 0, values["FWHM_4"], values["C4"])+
            gaussian(x, values["A5"], 0, values["FWHM_5"], values["C5"])+
            gaussian(x, values["A6"], 0, values["FWHM_6"], values["C6"])+
            gaussian(x, values["A7"], 0, values["FWHM_7"], values["C7"])+ 
            gaussian(x, values["A8"], 0, values["FWHM_8"], values["C8"])
        )

    return wrapped_model

def do_model_fit(bins, counts, errors, PrintParams=False, View=False,
                 fixed_params={}):
    
    all_params = ["A1", "FWHM_1", "A2", "FWHM_2", "C2", "A3", "FWHM_3", "C3", "A4", "FWHM_4", "C4",
                   "A5", "FWHM_5", "C5",
                   "A6", "FWHM_6", "C6",
                   "A7", "FWHM_7", "C7",
                   "A8", "FWHM_8", "C8"]
    free_params = [p for p in all_params if p not in fixed_params]

    # Default initial guesses
    default_guesses = {
        "A1": 51938.32,
        "FWHM_1": 72.81,
        "A2": 3600,
        "FWHM_2": 80,
        "C2": 173,
        "A3": 1920.24,
        "FWHM_3":74.8,
        "C3" : -88,
        "A4": 765.11,
        "FWHM_4" :74.26,
        "C4" : -314.44,
        "A5": 105,
        "FWHM_5": 96.41,
        "C5": 492.98,
        "A6": 32.43,
        "FWHM_6": 64.58,
        "C6": -505.76,
        "A7": 429,
        "FWHM_7": 170,
        "C7": -250,
        "A8": 1000,
        "FWHM_8": 65,
        "C8": -250
    }

    guess = [default_guesses[p] for p in free_params]
    lower_bounds = [-600] * len(free_params)
    upper_bounds = [np.inf] * len(free_params)
    
    if len(bins) == len(counts):
        xdata = bins
    elif len(bins) == len(counts) + 1:
        xdata = (bins[:-1] + bins[1:]) / 2
    else:
        raise ValueError("Mismatch in bins and counts length")

    model = make_wrapped_model(fixed_params)

    try:
        popt, pcov = curve_fit(
            model, xdata, counts, p0=guess,
            bounds=(lower_bounds, upper_bounds),
            sigma=errors, absolute_sigma=True, max_nfev=10000
        )
    except RuntimeError as e:
        print("Fit failed:", e)
        return

    # Merge fitted values into full param list
    fit_results = {}
    uncertainties = {}
    i = 0
    for p in all_params:
        if p in fixed_params:
            fit_results[p] = fixed_params[p]
            uncertainties[p] = 0
        else:
            fit_results[p] = popt[i]
            uncertainties[p] = np.sqrt(pcov[i, i])
            i += 1

    # Chi-squared
    fit_values = model(xdata, *popt)
    chi_squared = np.sum(((counts - fit_values) / errors) ** 2)
    dof = len(counts) - len(popt)
    chi_squared_red = chi_squared / dof

    # Optional print
    if PrintParams:
        for p in all_params:
            val = fit_results[p]
            err = uncertainties[p]
            print(f"{p} = {val:.2f} ± {err:.2f}" if err else f"{p} = {val:.2f} (fixed)")
        print(f"Chi^2: {chi_squared:.2f}")
        print(f"Reduced Chi^2: {chi_squared_red:.2f}")

    if View:
        # Plot data + fit + residuals
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axs[0].errorbar(xdata, counts, yerr=errors, fmt='.', markersize=3, label='Data')
        axs[0].plot(xdata, fit_values, '--', label='Model')
        axs[0].legend()
        axs[0].set_ylabel("Counts")
        axs[1].plot(xdata, (counts - fit_values) / errors, '.', color='red', markersize=3)
        axs[1].axhline(0, color='gray', ls='--')
        axs[1].set_ylabel("Residuals")
        axs[1].set_xlabel("t")
        plt.tight_layout()
        plt.show()

    return fit_results, uncertainties, chi_squared, chi_squared_red
        
    
# %% Actual Script of the first part (First histogram of the HBT ones)





# --- Dynamically find path of the current script and look for .csv ---
current_dir = Path(__file__).resolve().parent
csv_files = list(current_dir.glob("*.csv"))

if len(csv_files) != 1:
    raise RuntimeError(f"Expected exactly one CSV file in folder, found {len(csv_files)}.")

csv_file = csv_files[0]

# --- Load data ---
df = pd.read_csv(csv_file, sep=";")

positions = df.iloc[:, 0].values

hist3 = df.iloc[:, 3].values
hist4 = df.iloc[:, 4].values

ehist3 = np.sqrt(hist3)
ehist4 = np.sqrt(hist4)


peak_index3 = np.argmax(hist3)
peak_index4 = np.argmax(hist4)

peak_position3 = positions[peak_index3]
peak_position4 = positions[peak_index4]

positions3 = positions.copy() - peak_position3
positions4 = positions.copy() - peak_position4


arr = np.linspace(-1000, 1000, 15000)
# y1 = gaussian(arr, 55190, 0, 65, 0)
# y2 = sech2(arr, 55190, 0, 38, 0)
# y3 = y2 + gaussian(arr, 3550, 0, 80, 173)
# y4 = y2 + sech2(arr, 3550, 0, 80/1.76, 173)

# Define fixed params outside to reuse in both fit and plotting

# fixed_params = {"C8":-250}
fixed_params = {}
# fixed_params = {
#     "A1": 51938.32,
#     "FWHM_1": 72.81,
#     "A2": 3600,
#     "FWHM_2": 80,
#     "C2": 173,
#     "A3": 1920.24,
#     "FWHM_3":74.8,
#     "C3" : -88,
    
#     "A4": 947.60,
#     "FWHM_4" :120.28,
#     "C4" : -291.99,
#     "A5": 105,
#     "FWHM_5": 96.41,
#     "C5": 492.98,
#     "A6": 32.43,
#     "FWHM_6": 64.58,
#     "C6": -505.76
# }

# Run the fit
fit_results, errors, chi2, red_chi2 = do_model_fit(
    positions3, hist3, ehist3 + 0.001,
    fixed_params=fixed_params,
    PrintParams=True,
    View=True
)


# Define all parameter names involved in the model
param_names = ["A1", "FWHM_1", "A2", "FWHM_2", "C2", "A3", "FWHM_3", "C3", "A4", "FWHM_4", "C4",
               "A5", "FWHM_5", "C5",
               "A6", "FWHM_6", "C6",
               "A7", "FWHM_7", "C7",
               "A8", "FWHM_8", "C8"]

# Then extract the list of free (non-fixed) parameters
free_params = [p for p in param_names if p not in fixed_params]


# Combine fixed and fitted parameters into a single dictionary
all_params = {}
all_params.update(fixed_params)
all_params.update({p: fit_results[p] for p in free_params})


y1 = sech2(arr, all_params["A1"], 0, all_params["FWHM_1"]/1.76, 0)
y2 = sech2(arr, all_params["A2"], 0, all_params["FWHM_2"]/1.76, all_params["C2"])
y3 = gaussian(arr, all_params["A3"], 0, all_params["FWHM_3"], all_params["C3"])
y4 = gaussian(arr, all_params["A4"], 0, all_params["FWHM_4"], all_params["C4"])
y5 = gaussian(arr, all_params["A5"], 0, all_params["FWHM_5"], all_params["C5"])
y6 = gaussian(arr, all_params["A6"], 0, all_params["FWHM_6"], all_params["C6"])
y7 = gaussian(arr, all_params["A7"], 0, all_params["FWHM_7"], all_params["C7"])
y8 = gaussian(arr, all_params["A8"], 0, all_params["FWHM_8"], all_params["C8"])

#Plotting
plt.figure(figsize=(18, 8))
plt.errorbar(positions3, hist3, yerr=ehist3, fmt='.', color='green',
             capsize=2, ecolor='orange', label='HBT Detector2 vs Detector3', markersize = 4)

# Generate model prediction
model_curve = make_wrapped_model(fixed_params)
# free_params = [p for p in ["A1", "FWHM_1", "A2", "FWHM_2", "C2", "A3", "FWHM_3", "C3"] if p not in fixed_params]
x_fit = positions3
y_fit = model_curve(x_fit, *[fit_results[p] for p in free_params])

plt.plot(x_fit, y_fit, color='red', label='Model fit', ls='-.')


# Components
plt.plot(arr, y1, ls='--', color='blue', label='Component: sech² 1')
plt.plot(arr, y2, ls='--', color='purple', label='Component: sech² 2')
plt.plot(arr, y3, ls='--', color='gray', label='Component: Gaussian 1')
plt.plot(arr, y4, ls='--', color='yellow', label='Component: Gaussian 2')
plt.plot(arr, y5, ls='--', color='brown', label='Component: Gaussian 3')
plt.plot(arr, y6, ls='--', color='black', label='Component: Gaussian 4')
plt.plot(arr, y7, ls='--', color='#92C6D8', label='Component: Gaussian 5')
plt.plot(arr, y8, ls='-', color='#8B6AD2', label='Component: Gaussian 6')

#'lightblue'

plt.xlim(-580, 580)

# # Now safe to apply log scale!
# plt.yscale('log')
# plt.ylim(0.29, max(hist3) * 1.8)


plt.ylabel("Counts (a.u.)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% Compute the FWHM of the main peak of the model fitted to the datapoints
from scipy.signal import find_peaks

def compute_fwhm(x, y):
    half_max = np.max(y) / 2
    indices = np.where(y >= half_max)[0]

    if len(indices) < 2:
        return None  # Can't define FWHM

    x1 = x[indices[0]]
    x2 = x[indices[-1]]
    return x2 - x1

# Compute model fit over fine array
model_curve_total = model_curve(arr, *[fit_results[p] for p in free_params])

#Computing the FWHM of the main peak
fwhm_total = compute_fwhm(arr, model_curve_total)
print(f"FWHM of total model curve: {fwhm_total:.2f}", 2*'\n')

# Computing the positions of every peak from the model fit
# Find indices of peaks in the model curve
peak_indices, _ = find_peaks(model_curve_total, height=np.max(model_curve_total)*0.0004)  # 5% threshold

# Extract x positions of those peaks
peak_positions = arr[peak_indices]
peak_heights = model_curve_total[peak_indices]

# Display results
for i, (pos, height) in enumerate(zip(peak_positions, peak_heights), 1):
    print(f"Peak {i}: Position = {pos:.2f}, Height = {height:.2f}")

peak_data1 = np.column_stack((peak_positions, peak_heights))




plt.figure(figsize = (8,5))
plt.plot(arr, model_curve_total, label="Model")
plt.plot(peak_positions, peak_heights, '*', label='Detected Peaks', markersize = 10, color = 'red')
plt.xlabel("Position", fontsize = 16)
plt.ylabel("Counts", fontsize = 16)
plt.title("Detected Peaks", fontsize = 16)
plt.legend()
plt.yscale('log')
plt.ylim(4, max(hist3) * 1.8)

plt.grid(True)

# Customize ticks
plt.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.5)
plt.tick_params(axis='both', which='minor', labelsize=16, length=4, width=1)


plt.tight_layout()
plt.show()
# %% Salvataggio
import csv

# --- Save fit results ---
output_file = "fit_results06162025.csv"

# Combine parameter values and uncertainties into rows
rows = []
for param in param_names:
    value = fit_results[param]
    uncertainty = errors[param]
    status = "fixed" if param in fixed_params else "fitted"
    rows.append([param, value, uncertainty, status])

# Write to CSV
with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Parameter", "Value", "Uncertainty", "Status"])
    writer.writerows(rows)

print(f"Fit results saved to {output_file}")



# %% Fit of the other HBT histogram

def do_model_fit(bins, counts, errors, PrintParams=False, View=False,
                 fixed_params={}):
    
    all_params = ["A1", "FWHM_1", "A2", "FWHM_2", "C2", "A3", "FWHM_3", "C3", "A4", "FWHM_4", "C4",
                   "A5", "FWHM_5", "C5",
                   "A6", "FWHM_6", "C6",
                   "A7", "FWHM_7", "C7",
                   "A8", "FWHM_8", "C8"]
    free_params = [p for p in all_params if p not in fixed_params]

    # Default initial guesses
    default_guesses = {
        "A1": 51938.32,
        "FWHM_1": 72.81,
        "A2": 3600,
        "FWHM_2": 80,
        "C2": -173,
        "A3": 1920.24,
        "FWHM_3":74.8,
        "C3" : 88,
        "A4": 765.11,
        "FWHM_4" :74.26,
        "C4" : 314.44,
        "A5": 105,
        "FWHM_5": 96.41,
        "C5": -492.98,
        "A6": 32.43,
        "FWHM_6": 64.58,
        "C6": 505.76,
        "A7": 429,
        "FWHM_7": 170,
        "C7": 250,
        "A8": 1000,
        "FWHM_8": 65,
        "C8": 250
    }

    guess = [default_guesses[p] for p in free_params]
    lower_bounds = [-600] * len(free_params)
    upper_bounds = [np.inf] * len(free_params)
    
    if len(bins) == len(counts):
        xdata = bins
    elif len(bins) == len(counts) + 1:
        xdata = (bins[:-1] + bins[1:]) / 2
    else:
        raise ValueError("Mismatch in bins and counts length")

    model = make_wrapped_model(fixed_params)

    try:
        popt, pcov = curve_fit(
            model, xdata, counts, p0=guess,
            bounds=(lower_bounds, upper_bounds),
            sigma=errors, absolute_sigma=True, max_nfev=10000
        )
    except RuntimeError as e:
        print("Fit failed:", e)
        return

    # Merge fitted values into full param list
    fit_results = {}
    uncertainties = {}
    i = 0
    for p in all_params:
        if p in fixed_params:
            fit_results[p] = fixed_params[p]
            uncertainties[p] = 0
        else:
            fit_results[p] = popt[i]
            uncertainties[p] = np.sqrt(pcov[i, i])
            i += 1

    # Chi-squared
    fit_values = model(xdata, *popt)
    chi_squared = np.sum(((counts - fit_values) / errors) ** 2)
    dof = len(counts) - len(popt)
    chi_squared_red = chi_squared / dof

    # Optional print
    if PrintParams:
        for p in all_params:
            val = fit_results[p]
            err = uncertainties[p]
            print(f"{p} = {val:.2f} ± {err:.2f}" if err else f"{p} = {val:.2f} (fixed)")
        print(f"Chi^2: {chi_squared:.2f}")
        print(f"Reduced Chi^2: {chi_squared_red:.2f}")

    if View:
        # Plot data + fit + residuals
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axs[0].errorbar(xdata, counts, yerr=errors, fmt='.', markersize=3, label='Data')
        axs[0].plot(xdata, fit_values, '--', label='Model')
        axs[0].legend()
        axs[0].set_ylabel("Counts")
        axs[1].plot(xdata, (counts - fit_values) / errors, '.', color='red', markersize=3)
        axs[1].axhline(0, color='gray', ls='--')
        axs[1].set_ylabel("Residuals")
        axs[1].set_xlabel("t")
        plt.tight_layout()
        plt.show()

    return fit_results, uncertainties, chi_squared, chi_squared_red

# --- Dynamically find path of the current script and look for .csv ---
current_dir = Path(__file__).resolve().parent
csv_files = list(current_dir.glob("*.csv"))

if len(csv_files) != 1:
    raise RuntimeError(f"Expected exactly one CSV file in folder, found {len(csv_files)}.")

csv_file = csv_files[0]

# --- Load data ---
df = pd.read_csv(csv_file, sep=";")

positions = df.iloc[:, 0].values

hist3 = df.iloc[:, 3].values
hist4 = df.iloc[:, 4].values

ehist3 = np.sqrt(hist3)
ehist4 = np.sqrt(hist4)


peak_index3 = np.argmax(hist3)
peak_index4 = np.argmax(hist4)

peak_position3 = positions[peak_index3]
peak_position4 = positions[peak_index4]

positions3 = positions.copy() - peak_position3
positions4 = positions.copy() - peak_position4


arr = np.linspace(-1000, 1000, 15000)
# y1 = gaussian(arr, 55190, 0, 65, 0)
# y2 = sech2(arr, 55190, 0, 38, 0)
# y3 = y2 + gaussian(arr, 3550, 0, 80, 173)
# y4 = y2 + sech2(arr, 3550, 0, 80/1.76, 173)

# Define fixed params outside to reuse in both fit and plotting

# fixed_params = {"C8":-250}
fixed_params = {}
# fixed_params = {
#     "A1": 51938.32,
#     "FWHM_1": 72.81,
#     "A2": 3600,
#     "FWHM_2": 80,
#     "C2": 173,
#     "A3": 1920.24,
#     "FWHM_3":74.8,
#     "C3" : -88,
    
#     "A4": 947.60,
#     "FWHM_4" :120.28,
#     "C4" : -291.99,
#     "A5": 105,
#     "FWHM_5": 96.41,
#     "C5": 492.98,
#     "A6": 32.43,
#     "FWHM_6": 64.58,
#     "C6": -505.76
# }

# Run the fit
fit_results, errors, chi2, red_chi2 = do_model_fit(
    positions4, hist4, ehist4 + 0.001,
    fixed_params=fixed_params,
    PrintParams=True,
    View=True
)


# Define all parameter names involved in the model
param_names = ["A1", "FWHM_1", "A2", "FWHM_2", "C2", "A3", "FWHM_3", "C3", "A4", "FWHM_4", "C4",
               "A5", "FWHM_5", "C5",
               "A6", "FWHM_6", "C6",
               "A7", "FWHM_7", "C7",
               "A8", "FWHM_8", "C8"]

# Then extract the list of free (non-fixed) parameters
free_params = [p for p in param_names if p not in fixed_params]


# Combine fixed and fitted parameters into a single dictionary
all_params = {}
all_params.update(fixed_params)
all_params.update({p: fit_results[p] for p in free_params})


y1 = sech2(arr, all_params["A1"], 0, all_params["FWHM_1"]/1.76, 0)
y2 = sech2(arr, all_params["A2"], 0, all_params["FWHM_2"]/1.76, all_params["C2"])
y3 = gaussian(arr, all_params["A3"], 0, all_params["FWHM_3"], all_params["C3"])
y4 = gaussian(arr, all_params["A4"], 0, all_params["FWHM_4"], all_params["C4"])
y5 = gaussian(arr, all_params["A5"], 0, all_params["FWHM_5"], all_params["C5"])
y6 = gaussian(arr, all_params["A6"], 0, all_params["FWHM_6"], all_params["C6"])
y7 = gaussian(arr, all_params["A7"], 0, all_params["FWHM_7"], all_params["C7"])
y8 = gaussian(arr, all_params["A8"], 0, all_params["FWHM_8"], all_params["C8"])

#Plotting
plt.figure(figsize=(18, 8))
plt.errorbar(positions4, hist4, yerr=ehist3, fmt='.', color='green',
             capsize=2, ecolor='orange', label='HBT Detector3 vs Detector2', markersize = 4)

# Generate model prediction
model_curve = make_wrapped_model(fixed_params)
# free_params = [p for p in ["A1", "FWHM_1", "A2", "FWHM_2", "C2", "A3", "FWHM_3", "C3"] if p not in fixed_params]
x_fit = positions3
y_fit = model_curve(x_fit, *[fit_results[p] for p in free_params])

plt.plot(x_fit, y_fit, color='red', label='Model fit', ls='-.')


# Components
plt.plot(arr, y1, ls='--', color='blue', label='Component: sech² 1')
plt.plot(arr, y2, ls='--', color='purple', label='Component: sech² 2')
plt.plot(arr, y3, ls='--', color='gray', label='Component: Gaussian 1')
plt.plot(arr, y4, ls='--', color='yellow', label='Component: Gaussian 2')
plt.plot(arr, y5, ls='--', color='brown', label='Component: Gaussian 3')
plt.plot(arr, y6, ls='--', color='black', label='Component: Gaussian 4')
plt.plot(arr, y7, ls='--', color='#92C6D8', label='Component: Gaussian 5')
plt.plot(arr, y8, ls='-', color='#8B6AD2', label='Component: Gaussian 6')

#'lightblue'

plt.xlim(-580, 580)

# # Now safe to apply log scale!
plt.yscale('log')
plt.ylim(0.29, max(hist3) * 1.8)


plt.ylabel("Counts (a.u.)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Piccole informazioni aggiuntive (Few useful tips)
"""
To Access single categories of fit parameters you shouls use an instruction like this one !
a_values_sorted = [fit_results[f"A{i}"] for i in range(1, 9) if f"A{i}" in fit_results]
"""


# %% Compute the FWHM of the main peak of the model fitted to the datapoints
from scipy.signal import find_peaks

def compute_fwhm(x, y):
    half_max = np.max(y) / 2
    indices = np.where(y >= half_max)[0]

    if len(indices) < 2:
        return None  # Can't define FWHM

    x1 = x[indices[0]]
    x2 = x[indices[-1]]
    return x2 - x1

# Compute model fit over fine array
model_curve_total = model_curve(arr, *[fit_results[p] for p in free_params])

#Computing the FWHM of the main peak
fwhm_total = compute_fwhm(arr, model_curve_total)
print(f"FWHM of total model curve: {fwhm_total:.2f}", 2*'\n')

# Computing the positions of every peak from the model fit
# Find indices of peaks in the model curve
peak_indices, _ = find_peaks(model_curve_total, height=np.max(model_curve_total)*0.0004)  # 5% threshold

# Extract x positions of those peaks
peak_positions = arr[peak_indices]
peak_heights = model_curve_total[peak_indices]

# Display results
for i, (pos, height) in enumerate(zip(peak_positions, peak_heights), 1):
    print(f"Peak {i}: Position = {pos:.2f}, Height = {height:.2f}")

peak_data2 = np.column_stack((peak_positions, peak_heights))

plt.figure(figsize = (8,5))
plt.plot(arr, model_curve_total, label="Model")
plt.plot(peak_positions, peak_heights, 'rx', label='Detected Peaks')
plt.xlabel("Position")
plt.ylabel("Model Counts")
plt.title("Detected Peaks in Full Model")
plt.legend()
plt.yscale('log')
plt.ylim(0.29, max(hist3) * 1.8)
plt.grid(True)
plt.tight_layout()
plt.show()
# %% Plottaggio delle posizioni dei picchi rilevati in entrambi gli istogrammi di HBT per poter farne una correlazione !!


#Green Fluo : #2AE519


plt.figure(figsize=(10,6))
plt.scatter(peak_data1[:, 0], peak_data1[:, 1], marker='*', label = 'HBT Detector2 vs Detector3', color = 'Red', s = 75)  # x = positions, y = heights
# plt.scatter(peak_data2[:, 0], peak_data2[:, 1], marker='o', label = 'HBT Detector3 vs Detector2', color = '#40DCFF')
# plt.scatter(-peak_data1[:, 0], peak_data1[:, 1], marker='*', label = 'HBT Detector2 vs Detector3 (Time reverted)', color = '#E4080A', s = 5)

plt.xlabel("Position [t]", fontsize=16)
plt.ylabel("Counts", fontsize=16)
plt.title("Detected Peaks in HBT Histogram", fontsize = 16)

# Customize ticks
plt.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.5)
plt.tick_params(axis='both', which='minor', labelsize=16, length=4, width=1)

# Set other plot settings
plt.yscale('log')
plt.ylim(0.2, 1e6)
plt.xlim(-800, 800)

# Legend with larger font
# plt.legend(fontsize=10,           # Bigger text
#            handlelength=3,        # Length of the legend lines
#            handleheight=2,        # Height of the legend handles (vertical space)
#            handletextpad=1.5,     # Space between handle and text
#            markerscale=2)

# plt.grid(True)
plt.tight_layout()
plt.show()




# %% Only one function

def make_wrapped_model(fixed_params):
    param_names = ["A1", "FWHM_1", "A2", "FWHM_2", "C2"]

    def wrapped_model(x, *free_values):
        values = {}
        i = 0
        for name in param_names:
            if name in fixed_params:
                values[name] = fixed_params[name]
            else:
                values[name] = free_values[i]
                i += 1

        return (
            gaussian(x, values["A1"], 0, values["FWHM_1"], 0) +
            gaussian(x, values["A2"], 0, values["FWHM_2"], values["C2"])
        )

    return wrapped_model

def do_model_fit(bins, counts, errors, PrintParams=False, View=False,
                 fixed_params={}):
    
    all_params = ["A1", "FWHM_1", "A2", "FWHM_2", "C2"]
    free_params = [p for p in all_params if p not in fixed_params]

    # Default initial guesses
    default_guesses = {
        "A1": 51938.32,
        "FWHM_1": 72.81,
        "A2": 3600,
        "FWHM_2": 80,
        "C2": 173
    }

    guess = [default_guesses[p] for p in free_params]
    lower_bounds = [-600] * len(free_params)
    upper_bounds = [np.inf] * len(free_params)
    
    if len(bins) == len(counts):
        xdata = bins
    elif len(bins) == len(counts) + 1:
        xdata = (bins[:-1] + bins[1:]) / 2
    else:
        raise ValueError("Mismatch in bins and counts length")

    model = make_wrapped_model(fixed_params)

    try:
        popt, pcov = curve_fit(
            model, xdata, counts, p0=guess,
            bounds=(lower_bounds, upper_bounds),
            sigma=errors, absolute_sigma=True, max_nfev=10000
        )
    except RuntimeError as e:
        print("Fit failed:", e)
        return

    # Merge fitted values into full param list
    fit_results = {}
    uncertainties = {}
    i = 0
    for p in all_params:
        if p in fixed_params:
            fit_results[p] = fixed_params[p]
            uncertainties[p] = 0
        else:
            fit_results[p] = popt[i]
            uncertainties[p] = np.sqrt(pcov[i, i])
            i += 1

    # Chi-squared
    fit_values = model(xdata, *popt)
    chi_squared = np.sum(((counts - fit_values) / errors) ** 2)
    dof = len(counts) - len(popt)
    chi_squared_red = chi_squared / dof

    # Optional print
    if PrintParams:
        for p in all_params:
            val = fit_results[p]
            err = uncertainties[p]
            print(f"{p} = {val:.2f} ± {err:.2f}" if err else f"{p} = {val:.2f} (fixed)")
        print(f"Chi^2: {chi_squared:.2f}")
        print(f"Reduced Chi^2: {chi_squared_red:.2f}")

    if View:
        # Plot data + fit + residuals
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axs[0].errorbar(xdata, counts, yerr=errors, fmt='.', markersize=3, label='Data')
        axs[0].plot(xdata, fit_values, '--', label='Model')
        axs[0].legend()
        axs[0].set_ylabel("Counts")
        axs[1].plot(xdata, (counts - fit_values) / errors, '.', color='red', markersize=3)
        axs[1].axhline(0, color='gray', ls='--')
        axs[1].set_ylabel("Residuals")
        axs[1].set_xlabel("t")
        plt.tight_layout()
        plt.show()

    return fit_results, uncertainties, chi_squared, chi_squared_red
        


# --- Dynamically find path of the current script and look for .csv ---
current_dir = Path(__file__).resolve().parent
csv_files = list(current_dir.glob("*.csv"))

if len(csv_files) != 1:
    raise RuntimeError(f"Expected exactly one CSV file in folder, found {len(csv_files)}.")

csv_file = csv_files[0]

# --- Load data ---
df = pd.read_csv(csv_file, sep=";")

positions = df.iloc[:, 0].values

hist3 = df.iloc[:, 3].values
hist4 = df.iloc[:, 4].values

ehist3 = np.sqrt(hist3)
ehist4 = np.sqrt(hist4)


peak_index3 = np.argmax(hist3)
peak_index4 = np.argmax(hist4)

peak_position3 = positions[peak_index3]
peak_position4 = positions[peak_index4]

positions3 = positions.copy() - peak_position3
positions4 = positions.copy() - peak_position4


arr = np.linspace(-1000, 1000, 15000)
# y1 = gaussian(arr, 55190, 0, 65, 0)
# y2 = sech2(arr, 55190, 0, 38, 0)
# y3 = y2 + gaussian(arr, 3550, 0, 80, 173)
# y4 = y2 + sech2(arr, 3550, 0, 80/1.76, 173)

# Define fixed params outside to reuse in both fit and plotting

# fixed_params = {"C8":-250}
fixed_params = {}
# fixed_params = {
#     "A1": 51938.32,
#     "FWHM_1": 72.81,
#     "A2": 3600,
#     "FWHM_2": 80,
#     "C2": 173,
#     "A3": 1920.24,
#     "FWHM_3":74.8,
#     "C3" : -88,
    
#     "A4": 947.60,
#     "FWHM_4" :120.28,
#     "C4" : -291.99,
#     "A5": 105,
#     "FWHM_5": 96.41,
#     "C5": 492.98,
#     "A6": 32.43,
#     "FWHM_6": 64.58,
#     "C6": -505.76
# }

# Run the fit
fit_results, errors, chi2, red_chi2 = do_model_fit(
    positions3, hist3, ehist3 + 0.001,
    fixed_params=fixed_params,
    PrintParams=True,
    View=True
)


# Define all parameter names involved in the model
param_names = ["A1", "FWHM_1", "A2", "FWHM_2", "C2"]

# Then extract the list of free (non-fixed) parameters
free_params = [p for p in param_names if p not in fixed_params]


# Combine fixed and fitted parameters into a single dictionary
all_params = {}
all_params.update(fixed_params)
all_params.update({p: fit_results[p] for p in free_params})


y1 = gaussian(arr, all_params["A1"], 0, all_params["FWHM_1"], 0)
y2 = gaussian(arr, all_params["A2"], 0, all_params["FWHM_2"], all_params["C2"])
# y3 = gaussian(arr, all_params["A3"], 0, all_params["FWHM_3"], all_params["C3"])


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

x_fine3, y_fine3, xL3, xR3, hmax3, fwhm3 = compute_fwhm(positions3, hist3)

from scipy.signal import find_peaks

def compute_fwhm(x, y):
    half_max = np.max(y) / 2
    indices = np.where(y >= half_max)[0]

    if len(indices) < 2:
        return None  # Can't define FWHM

    x1 = x[indices[0]]
    x2 = x[indices[-1]]
    return x2 - x1

# Compute model fit over fine array
model_curve_total = model_curve(arr, *[fit_results[p] for p in free_params])

#Computing the FWHM of the main peak
fwhm_total = compute_fwhm(arr, model_curve_total)
print(f"FWHM of total model curve: {fwhm_total:.2f}", 2*'\n')

#Plotting
plt.figure(figsize=(18, 8))
plt.errorbar(positions3, hist3, yerr=ehist3, fmt='.', color='green',
             capsize=2, ecolor='orange', label='HBT Detector2 vs Detector3', markersize = 4)

# Generate model prediction
model_curve = make_wrapped_model(fixed_params)
# free_params = [p for p in ["A1", "FWHM_1", "A2", "FWHM_2", "C2", "A3", "FWHM_3", "C3"] if p not in fixed_params]
x_fit = positions3
y_fit = model_curve(x_fit, *[fit_results[p] for p in free_params])

plt.plot(x_fit, y_fit, color='red', label='Model fit', ls='-.')


# Components
plt.plot(arr, y1, ls='--', color='blue', label='Component: Gaussian 1')
plt.plot(arr, y2, ls='--', color='purple', label='Component: Gaussian 2')
# plt.plot(arr, y3, ls='--', color='gray', label='Component: Gaussian 3')

plt.hlines(hmax3, xL3, xR3, colors='green', linewidth=2,
           label=f'FWHM HBT = {fwhm_total:.2f}', alpha=0.7)

plt.text(50, 361, f'{fwhm_total:.2f}ps', color='Green', ha='center')

#'lightblue'

plt.xlim(-580, 580)

# # Now safe to apply log scale!
# plt.yscale('log')
# plt.ylim(0.29, max(hist3) * 1.8)


plt.ylabel("Counts (a.u.)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% Altro istogramma

"""
Non basta solo una funzione, pare che ci sia qualcosa di strano sotto...
Devi provare anche con 3 funzioni sotto per poter ottimizzare il tutto...

"""

def make_wrapped_model(fixed_params):
    param_names = ["A1", "FWHM_1"]

    def wrapped_model(x, *free_values):
        values = {}
        i = 0
        for name in param_names:
            if name in fixed_params:
                values[name] = fixed_params[name]
            else:
                values[name] = free_values[i]
                i += 1

        return (
            sech2(x, values["A1"], 0, values["FWHM_1"]/1.76, 0))
        

    return wrapped_model

def do_model_fit(bins, counts, errors, PrintParams=False, View=False,
                 fixed_params={}):
    
    all_params = ["A1", "FWHM_1"]
    free_params = [p for p in all_params if p not in fixed_params]

    # Default initial guesses
    default_guesses = {
        "A1": 750.32,
        "FWHM_1": 50
    }

    guess = [default_guesses[p] for p in free_params]
    lower_bounds = [-600] * len(free_params)
    upper_bounds = [np.inf] * len(free_params)
    
    if len(bins) == len(counts):
        xdata = bins
    elif len(bins) == len(counts) + 1:
        xdata = (bins[:-1] + bins[1:]) / 2
    else:
        raise ValueError("Mismatch in bins and counts length")

    model = make_wrapped_model(fixed_params)

    try:
        popt, pcov = curve_fit(
            model, xdata, counts, p0=guess,
            bounds=(lower_bounds, upper_bounds),
            sigma=errors, absolute_sigma=True, max_nfev=10000
        )
    except RuntimeError as e:
        print("Fit failed:", e)
        return

    # Merge fitted values into full param list
    fit_results = {}
    uncertainties = {}
    i = 0
    for p in all_params:
        if p in fixed_params:
            fit_results[p] = fixed_params[p]
            uncertainties[p] = 0
        else:
            fit_results[p] = popt[i]
            uncertainties[p] = np.sqrt(pcov[i, i])
            i += 1

    # Chi-squared
    fit_values = model(xdata, *popt)
    chi_squared = np.sum(((counts - fit_values) / errors) ** 2)
    dof = len(counts) - len(popt)
    chi_squared_red = chi_squared / dof

    # Optional print
    if PrintParams:
        for p in all_params:
            val = fit_results[p]
            err = uncertainties[p]
            print(f"{p} = {val:.2f} ± {err:.2f}" if err else f"{p} = {val:.2f} (fixed)")
        print(f"Chi^2: {chi_squared:.2f}")
        print(f"Reduced Chi^2: {chi_squared_red:.2f}")

    if View:
        # Plot data + fit + residuals
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axs[0].errorbar(xdata, counts, yerr=errors, fmt='.', markersize=3, label='Data')
        axs[0].plot(xdata, fit_values, '--', label='Model')
        axs[0].legend()
        axs[0].set_ylabel("Counts")
        axs[1].plot(xdata, (counts - fit_values) / errors, '.', color='red', markersize=3)
        axs[1].axhline(0, color='gray', ls='--')
        axs[1].set_ylabel("Residuals")
        axs[1].set_xlabel("t")
        plt.tight_layout()
        plt.show()

    return fit_results, uncertainties, chi_squared, chi_squared_red
        

# --- Dynamically find path of the current script and look for .csv ---
current_dir = Path(__file__).resolve().parent
csv_files = list(current_dir.glob("*.csv"))

if len(csv_files) != 1:
    raise RuntimeError(f"Expected exactly one CSV file in folder, found {len(csv_files)}.")

csv_file = csv_files[0]

# --- Load data ---
df = pd.read_csv(csv_file, sep=";")

positions = df.iloc[:, 0].values

hist3 = df.iloc[:, 3].values
hist4 = df.iloc[:, 4].values

ehist3 = np.sqrt(hist3)
ehist4 = np.sqrt(hist4)


peak_index3 = np.argmax(hist3)
peak_index4 = np.argmax(hist4)

peak_position3 = positions[peak_index3]
peak_position4 = positions[peak_index4]

positions3 = positions.copy() - peak_position3
positions4 = positions.copy() - peak_position4



selection_mask = (positions > -1000) & (positions < 1000)
#Restricting the arrays to the current interval being studied !

positions4 = positions4[selection_mask]


hist4 = hist4[selection_mask]


ehist4 = ehist4[selection_mask]


arr = np.linspace(-1000, 1000, 15000)
# y1 = gaussian(arr, 55190, 0, 65, 0)
# y2 = sech2(arr, 55190, 0, 38, 0)
# y3 = y2 + gaussian(arr, 3550, 0, 80, 173)
# y4 = y2 + sech2(arr, 3550, 0, 80/1.76, 173)

# Define fixed params outside to reuse in both fit and plotting

# fixed_params = {"C8":-250}
# fixed_params = {}
fixed_params = {
    # "A1": 780.32}
    # "FWHM_1": 72.81,
#     "A2": 3600,
#     "FWHM_2": 80,
#     "C2": 173,
#     "A3": 1920.24,
#     "FWHM_3":74.8,
#     "C3" : -88,
    
#     "A4": 947.60,
#     "FWHM_4" :120.28,
#     "C4" : -291.99,
#     "A5": 105,
#     "FWHM_5": 96.41,
#     "C5": 492.98,
#     "A6": 32.43,
#     "FWHM_6": 64.58,
#     "C6": -505.76
}

# Run the fit
fit_results, errors, chi2, red_chi2 = do_model_fit(
    positions4, hist4, ehist4 + 0.0001,
    fixed_params=fixed_params,
    PrintParams=True,
    View=True
)


# Define all parameter names involved in the model
param_names = ["A1", "FWHM_1"]

# Then extract the list of free (non-fixed) parameters
free_params = [p for p in param_names if p not in fixed_params]


# Combine fixed and fitted parameters into a single dictionary
all_params = {}
all_params.update(fixed_params)
all_params.update({p: fit_results[p] for p in free_params})


y1 = gaussian(arr, all_params["A1"], 0, all_params["FWHM_1"], 0)
# y2 = gaussian(arr, all_params["A2"], 0, all_params["FWHM_2"], all_params["C2"])
# y3 = gaussian(arr, all_params["A3"], 0, all_params["FWHM_3"], all_params["C3"])


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

x_fine3, y_fine3, xL3, xR3, hmax3, fwhm3 = compute_fwhm(positions3, hist3)

from scipy.signal import find_peaks

def compute_fwhm(x, y):
    half_max = np.max(y) / 2
    indices = np.where(y >= half_max)[0]

    if len(indices) < 2:
        return None  # Can't define FWHM

    x1 = x[indices[0]]
    x2 = x[indices[-1]]
    return x2 - x1

# Compute model fit over fine array

model_curve = make_wrapped_model(fixed_params)

model_curve_total = model_curve(arr, *[fit_results[p] for p in free_params])

#Computing the FWHM of the main peak
fwhm_total = compute_fwhm(arr, model_curve_total)
print(f"FWHM of total model curve: {fwhm_total:.2f}", 2*'\n')

#Plotting
plt.figure(figsize=(18, 8))
plt.errorbar(positions4, hist4, yerr=ehist4, fmt='.', color='green',
             capsize=2, ecolor='orange', label='HBT Detector2 vs Detector3', markersize = 4)

# Generate model prediction
model_curve = make_wrapped_model(fixed_params)
# free_params = [p for p in ["A1", "FWHM_1", "A2", "FWHM_2", "C2", "A3", "FWHM_3", "C3"] if p not in fixed_params]
x_fit = positions4
y_fit = model_curve(x_fit, *[fit_results[p] for p in free_params])

plt.plot(x_fit, y_fit, color='red', label='Model fit', ls='-.')


# Components
plt.plot(arr, y1, ls='--', color='blue', label='Component: Gaussian 1')
# plt.plot(arr, y2, ls='--', color='purple', label='Component: Gaussian 2')
# plt.plot(arr, y3, ls='--', color='gray', label='Component: Gaussian 3')

# plt.hlines(hmax3, xL3, xR3, colors='green', linewidth=2,
#            label=f'FWHM HBT = {fwhm_total:.2f}', alpha=0.7)

# plt.text(50, 361, f'{fwhm_total:.2f}ps', color='Green', ha='center')

#'lightblue'

plt.xlim(-580, 580)

# # Now safe to apply log scale!
# plt.yscale('log')
# plt.ylim(0.29, max(hist3) * 1.8)


plt.ylabel("Counts (a.u.)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


