#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:23:05 2025

@author: andreamaccarinelli
"""

"""
import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma = 1.0  # Pulse width
sigma_j = 0.5  # Temporal jitter standard deviation
T_rep = 5.0  # Pulse repetition period
num_pulses = 100  # Number of pulses in the train
time_range = 50  # Total time window

# Generate pulse centers with jitter
pulse_centers = np.arange(-time_range, time_range, T_rep)  # Ideal pulse positions
jitter = np.random.normal(0, sigma_j, size=pulse_centers.shape)  # Random shifts
pulse_centers_with_jitter = pulse_centers + jitter  # Apply jitter

# Define time axis
t = np.linspace(-time_range, time_range, 5000)

# Construct total electric field with and without jitter
E_total_no_jitter = np.zeros_like(t)
E_total_with_jitter = np.zeros_like(t)

for t_i in pulse_centers:
    E_total_no_jitter += np.exp(-(t - t_i)**2 / (4 * sigma**2))  # Gaussian field without jitter

for t_i in pulse_centers_with_jitter:
    E_total_with_jitter += np.exp(-(t - t_i)**2 / (4 * sigma**2))  # Gaussian field with jitter

# Compute intensity profiles: I(t) = |E_total|^2
I_total_no_jitter = E_total_no_jitter**2
I_total_with_jitter = E_total_with_jitter**2

# Compute g(1)(tau) using intensity (with jitter)
taus = np.linspace(-10, 10, 300)  # Delay values
g1_tau = np.array([
    np.mean(I_total_with_jitter * np.interp(t - tau, t, I_total_with_jitter)) for tau in taus
])
g1_tau /= np.mean(I_total_with_jitter)**2  # Normalize by <I>^2

# Ideal g(1)(tau) without jitter (for comparison)
g1_tau_ideal = np.exp(-taus**2 / (4 * sigma**2))  # Since g(1) now deals with intensity, the exponent is different

# Plot results
plt.figure(figsize=(10, 6))

# Plot g(1)(tau)
plt.plot(taus, g1_tau, label="With Jitter", color='blue', linewidth=2)
plt.plot(taus, g1_tau_ideal, label="Ideal (No Jitter)", linestyle="--", color='red', linewidth=2)

# Plot intensity profiles
plt.plot(t, I_total_no_jitter, label="Intensity Profile without Jitter", color='green', linestyle='-.', linewidth=2)
plt.plot(t, I_total_with_jitter, label="Intensity Profile with Jitter", color='orange', linestyle=':', linewidth=2)

# Labels and legend
plt.title(r'First-Order Correlation Function $g^{(1)}(\tau)$ with Temporal Jitter (Intensity-Based)', fontsize=14)
plt.xlabel(r'$\tau$', fontsize=12)
plt.ylabel(r'$g^{(1)}(\tau)$', fontsize=12)
plt.legend()

# Show grid and plot
plt.grid(True)
plt.xlim(-8, 8)
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma = 1.0  # Pulse width
sigma_j = 0.5  # Temporal jitter standard deviation
T_rep = 5.0  # Pulse repetition period
num_pulses = 100  # Number of pulses in the train
time_range = 50  # Total time window

# Generate pulse centers with jitter
pulse_centers = np.arange(-time_range, time_range, T_rep)  # Ideal pulse positions
jitter = np.random.normal(0, sigma_j, size=pulse_centers.shape)  # Random shifts
pulse_centers_with_jitter = pulse_centers + jitter  # Apply jitter

# Define time axis
t = np.linspace(-time_range, time_range, 5000)

# Construct total electric field with and without jitter
E_total_no_jitter = np.zeros_like(t)
E_total_with_jitter = np.zeros_like(t)

for t_i in pulse_centers:
    E_total_no_jitter += np.exp(-(t - t_i)**2 / (4 * sigma**2))  # Gaussian field without jitter

for t_i in pulse_centers_with_jitter:
    E_total_with_jitter += np.exp(-(t - t_i)**2 / (4 * sigma**2))  # Gaussian field with jitter

# Compute intensity profiles: I(t) = |E_total|^2
I_total_no_jitter = E_total_no_jitter**2
I_total_with_jitter = E_total_with_jitter**2

# Compute g(1)(tau) using intensity (with jitter)
taus = np.linspace(-10, 10, 300)  # Delay values
g1_tau = np.array([
    np.mean(I_total_with_jitter * np.interp(t - tau, t, I_total_with_jitter)) for tau in taus
])
g1_tau /= np.mean(I_total_with_jitter)**2  # Normalize by <I>^2

# Ideal g(1)(tau) without jitter (for comparison)
g1_tau_ideal = np.exp(-taus**2 / (4 * sigma**2))  # Since g(1) now deals with intensity, the exponent is different

# Plot results
plt.figure(figsize=(10, 6))

# Plot g(1)(tau)
plt.plot(taus, g1_tau, label="With Jitter", color='blue', linewidth=2)
plt.plot(taus, g1_tau_ideal, label="Ideal (No Jitter)", linestyle="--", color='red', linewidth=2)

# Plot intensity profiles
plt.plot(t, I_total_no_jitter, label="Intensity Profile without Jitter", color='green', linestyle='-.', linewidth=2)
plt.plot(t, I_total_with_jitter, label="Intensity Profile with Jitter", color='orange', linestyle=':', linewidth=2)

# Labels and legend
plt.title(r'Second-Order Correlation Function $g^{(2)}(\tau)$ with Temporal Jitter (Intensity-Based)', fontsize=14)
plt.xlabel(r'$\tau$', fontsize=12)
plt.ylabel(r'$g^{(2)}(\tau)$', fontsize=12)
plt.legend()

# Show grid and plot
plt.grid(True)
plt.xlim(-8, 8)  # Restrict x-axis range to [-8, 8]
plt.show()

# %%  Cross Correlation Functions for g1(\tau) at different distances in time 


def cross_correlation(reference_pulse, pulse, t, tau):
    """
    Compute the cross-correlation between the reference pulse and the pulse at time tau.

    Parameters:
    - reference_pulse: Pulse at time t=0 (reference pulse)
    - pulse: Pulse at time t-tau (generic pulse)
    - t: Time array
    - tau: Time shift

    Returns:
    - Cross-correlation value at each time
    """
    return np.mean(reference_pulse * np.interp(t - tau, t, pulse))

# Parameters
sigma = 1.0  # Pulse width
sigma_j = 0.5  # Temporal jitter standard deviation
T_rep = 5.0  # Pulse repetition period
num_pulses = 100  # Number of pulses in the train
time_range = 50  # Total time window

# Generate pulse centers with jitter
pulse_centers = np.arange(-time_range, time_range, T_rep)  # Ideal pulse positions
jitter = np.random.normal(0, sigma_j, size=pulse_centers.shape)  # Random shifts
pulse_centers_with_jitter = pulse_centers + jitter  # Apply jitter

# Define time axis
t = np.linspace(-time_range, time_range, 5000)

# Construct total electric field with jitter
E_total_with_jitter = np.zeros_like(t)
for t_i in pulse_centers_with_jitter:
    E_total_with_jitter += np.exp(-(t - t_i)**2 / (4 * sigma**2))  # Gaussian field with jitter

# Define the reference pulse (at t=0) for cross-correlation
reference_pulse = np.exp(-(t - 0)**2 / (4 * sigma**2))

# Compute cross-correlation at different time shifts (tau)
taus = np.linspace(-25, 25, 300)  # Delay values for cross-correlation
cross_corr_values = np.zeros_like(taus)

for idx, tau in enumerate(taus):
    cross_corr_values[idx] = cross_correlation(reference_pulse, E_total_with_jitter, t, tau)

# Normalize the cross-correlation by the maximum value
cross_corr_values /= np.max(np.abs(cross_corr_values))

# Plotting the cross-correlation for different tau values
plt.figure(figsize=(10, 6))

# Plot cross-correlation for different taus
plt.plot(taus, cross_corr_values, label=r'Cross-correlation between reference and pulse at $\tau$', color='blue', linewidth=2)

# Add labels and legend
plt.title(r'Cross-Correlation Function $C(\tau)$ between Reference Pulse and Other Pulses', fontsize=14)
plt.xlabel(r'$\tau$', fontsize=12)
plt.ylabel(r'$C(\tau)$', fontsize=12)
plt.legend()

# Show grid and plot
plt.grid(True)
plt.xlim(-20, 20)  # Restrict the x-axis as requested
plt.show()

# %% Cross Correlation Functions 2nd Version 


def intensity_cross_correlation(reference_intensity, intensity, t, tau):
    """
    Compute the cross-correlation of intensities between the reference intensity and a shifted pulse intensity.

    Parameters:
    - reference_intensity: Intensity profile of the reference pulse (at time t=0)
    - intensity: Intensity profile of the pulse at time t-tau
    - t: Time array
    - tau: Time shift

    Returns:
    - Cross-correlation value at each time
    """
    return np.mean(reference_intensity * np.interp(t - tau, t, intensity))



# Parameters
sigma = 1.0  # Pulse width
sigma_j = 0.5  # Temporal jitter standard deviation
T_rep = 5.0  # Pulse repetition period
num_pulses = 100  # Number of pulses in the train
time_range = 50  # Total time window

# Generate pulse centers with jitter
pulse_centers = np.arange(-time_range, time_range, T_rep)  # Ideal pulse positions
jitter = np.random.normal(0, sigma_j, size=pulse_centers.shape)  # Random shifts
pulse_centers_with_jitter = pulse_centers + jitter  # Apply jitter

# Define time axis
t = np.linspace(-time_range, time_range, 5000)

# Construct total electric field with jitter (without squaring yet)
E_total_with_jitter = np.zeros_like(t)
for t_i in pulse_centers_with_jitter:
    E_total_with_jitter += np.exp(-(t - t_i)**2 / (4 * sigma**2))  # Gaussian field with jitter

# Calculate intensity profile by squaring the electric field
I_total_with_jitter = E_total_with_jitter**2  # Intensity profile

# Define the reference pulse (at t=0) for cross-correlation
reference_pulse = np.exp(-(t - 0)**2 / (4 * sigma**2))
reference_intensity = reference_pulse**2  # Intensity profile of the reference pulse

# Compute cross-correlation of intensities at different time shifts (tau)
taus = np.linspace(-30, 30, 300)  # Delay values for cross-correlation
cross_corr_intensity_values = np.zeros_like(taus)

for idx, tau in enumerate(taus):
    cross_corr_intensity_values[idx] = intensity_cross_correlation(reference_intensity, I_total_with_jitter, t, tau)

# Normalize the cross-correlation by the maximum value
cross_corr_intensity_values /= np.max(np.abs(cross_corr_intensity_values))

# Plotting the cross-correlation for different tau values
plt.figure(figsize=(10, 6))

# Plot cross-correlation of intensities for different taus
plt.plot(taus, cross_corr_intensity_values, label=r'Intensity Cross-Correlation between reference and pulse at $\tau$', color='blue', linewidth=2)

# Add labels and legend
plt.title(r'Intensity-Based Cross-Correlation Function $g^{(2)}(\tau)$ between Reference Pulse and Other Pulses', fontsize=14)
plt.xlabel(r'$\tau$', fontsize=12)
plt.ylabel(r'$g^{(2)}(\tau)$', fontsize=12)
plt.legend()

# Show grid and plot
plt.grid(True)
plt.xlim(-40, 40)  # Restrict the x-axis as requested
plt.show()


# %% Test 3 

import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma = 1.0  # Pulse width
sigma_j_gen = 0.5  # Jitter in generation (temporal jitter standard deviation)
sigma_j_ps = 0.5  # Jitter in pulse shape (temporal jitter standard deviation)
T_rep = 5.0  # Pulse repetition period
num_pulses = 100  # Number of pulses in the train
time_range = 50  # Total time window
I_0 = 1.0  # Intensity scale factor

# Generate pulse centers with jitter (both generation jitter and pulse shape jitter)
pulse_centers = np.arange(-time_range, time_range, T_rep)  # Ideal pulse positions
jitter_gen = np.random.normal(0, sigma_j_gen, size=pulse_centers.shape)  # Generation jitter
jitter_ps = np.random.normal(0, sigma_j_ps, size=pulse_centers.shape)  # Pulse shape jitter
pulse_centers_with_jitter = pulse_centers + jitter_gen + jitter_ps  # Apply jitter

# Define time axis
t = np.linspace(-time_range, time_range, 5000)

# Construct intensity profiles at time t and t - tau
def intensity(t, pulse_centers, sigma):
    intensity_profile = np.zeros_like(t)
    for t_i in pulse_centers:
        intensity_profile += I_0 * np.exp(-(t - t_i)**2 / (2 * sigma**2))  # Gaussian field
    return intensity_profile ** 2  # Return intensity (squared field)

# Compute intensities for two pulse trains at t and t - tau
tau_vals = np.linspace(-100, 100, 300)  # Delay values for cross-correlation
cross_corr_vals = np.zeros_like(tau_vals)

for idx, tau in enumerate(tau_vals):
    I_t = intensity(t, pulse_centers_with_jitter, sigma)  # Intensity at time t
    I_t_minus_tau = intensity(t, pulse_centers_with_jitter - tau, sigma)  # Intensity at time t - tau
    cross_corr_vals[idx] = np.mean(I_t * I_t_minus_tau)  # Cross-correlation (product of intensities)

# Normalize the cross-correlation
cross_corr_vals /= np.max(np.abs(cross_corr_vals))

# Plot the cross-correlation function for various tau
plt.figure(figsize=(10, 6))
plt.plot(tau_vals, cross_corr_vals, label=r'Intensity Cross-Correlation', color='blue', linewidth=2)
plt.title(r'Intensity-Based Cross-Correlation Function $C_I(\tau)$ with Temporal Jitter', fontsize=14)
plt.xlabel(r'$\tau$', fontsize=12)
plt.ylabel(r'$C_I(\tau)$', fontsize=12)
plt.legend()
plt.grid(True)
plt.xlim(-100, +100)  # Restrict x-axis range to focus on the peaks
plt.show()


# %% Ciaone 2.1

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parameters
sigma = 1.0  # Pulse width
sigma_j_gen = 0.6  # Jitter in generation (temporal jitter standard deviation)
sigma_j_ps = 0.6  # Jitter in pulse shape (temporal jitter standard deviation)
T_rep = 5.0  # Pulse repetition period
num_pulses = 100  # Number of pulses in the train
time_range = 200  # Total time window
I_0 = 1.0  # Intensity scale factor

# Generate pulse centers with jitter (both generation jitter and pulse shape jitter)
pulse_centers = np.arange(-time_range, time_range, T_rep)  # Ideal pulse positions
jitter_gen = np.random.normal(0, sigma_j_gen, size=pulse_centers.shape)  # Generation jitter
jitter_ps = np.random.normal(0, sigma_j_ps, size=pulse_centers.shape)  # Pulse shape jitter
pulse_centers_with_jitter = pulse_centers + jitter_gen + jitter_ps  # Apply jitter

# Define time axis
t = np.linspace(-time_range, time_range, 5000)

# Construct intensity profiles at time t and t - tau
def intensity(t, pulse_centers, sigma):
    intensity_profile = np.zeros_like(t)
    for t_i in pulse_centers:
        intensity_profile += I_0 * np.exp(-(t - t_i)**2 / (2 * sigma**2))  # Gaussian field
    return intensity_profile ** 2  # Return intensity (squared field)

# Compute intensities for two pulse trains at t and t - tau
tau_vals = np.linspace(-3 * T_rep, 3 * T_rep, 500)  # Ridurre il numero di punti a 500 o 1000
cross_corr_vals = np.zeros_like(tau_vals)

for idx, tau in enumerate(tau_vals):
    I_t = intensity(t, pulse_centers_with_jitter, sigma)  # Intensity at time t
    I_t_minus_tau = intensity(t, pulse_centers_with_jitter - tau, sigma)  # Intensity at time t - tau
    cross_corr_vals[idx] = np.mean(I_t * I_t_minus_tau)  # Cross-correlation (product of intensities)

# Normalize the cross-correlation
cross_corr_vals /= np.max(np.abs(cross_corr_vals))

# Find the peaks in the cross-correlation
peaks, _ = find_peaks(cross_corr_vals)

# Initialize list to store FWHM values
fwhm_values = []

# For each peak, calculate the FWHM
for peak in peaks:
    # Get the maximum value at the peak
    max_value = cross_corr_vals[peak]
    
    # Calculate the half-maximum
    half_max = max_value / 2
    
    # Check for left side of the peak
    left_idx = np.where(cross_corr_vals[:peak] < half_max)[0]
    if len(left_idx) > 0:
        left_idx = left_idx[-1]  # Last index before the peak
    else:
        left_idx = 0  # If no crossing found, default to start of array
    
    # Check for right side of the peak
    right_idx = np.where(cross_corr_vals[peak:] < half_max)[0]
    if len(right_idx) > 0:
        right_idx = right_idx[0] + peak  # First index after the peak
    else:
        right_idx = len(cross_corr_vals) - 1  # If no crossing found, default to end of array
    
    # Calculate the FWHM as the difference in indices
    fwhm = t[right_idx] - t[left_idx]
    fwhm_values.append(fwhm)

# Plot FWHM as a function of tau
plt.figure(figsize=(10, 6))
plt.plot(tau_vals[peaks], fwhm_values, 'bo-', label='FWHM vs tau')
plt.title(r'FWHM of Cross-Correlation Peaks vs $\tau$', fontsize=14)
plt.xlabel(r'$\tau$', fontsize=12)
plt.ylabel('FWHM (Time)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

