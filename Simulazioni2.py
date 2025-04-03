#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 15:44:44 2025

@author: andreamaccarinelli
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma = 1.0  # pulse width
tau = np.linspace(-5, 5, 1000)  # time interval

# Enable LaTeX support in matplotlib
plt.rcParams['text.usetex'] = True  

# First-order correlation function g(1)(tau)
g1_tau = np.exp(-tau**2 / (8 * sigma**2))

# Gaussian pulse for the electric field (with 1/4sigma^2 in the exponent)
gaussian_field = np.exp(-tau**2 / (4 * sigma**2))

# Plotting
plt.figure(figsize=(10, 6))

# Plot g(1)(tau)
plt.plot(tau, g1_tau, label=r'$g^{(1)}(\tau) = e^{-\tau^2 / (8\sigma^2)}$', color='blue', linewidth=2)

# Plot Gaussian pulse of the electric field
plt.plot(tau, gaussian_field, label=r'$E(t) \sim e^{-t^2 / (4\sigma^2)}$', color='red', linestyle='--', linewidth=2)

# Labels and legend
plt.title(r'Comparison between $g^{(1)}(\tau)$ and the electric field profile', fontsize=14)
plt.xlabel(r'$\tau$', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.legend(fontsize = 16)

# Show the plot
plt.grid(True)
plt.show()
