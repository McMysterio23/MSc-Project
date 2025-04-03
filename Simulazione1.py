#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:17:24 2025

@author: andreamaccarinelli
"""

#Ciaone
import numpy as np
import matplotlib.pyplot as plt

def g1_no_jitter(tau, sigma):
    """Funzione di correlazione g^(1)(tau) senza jitter"""
    return np.exp(-tau**2 / (2 * sigma**2))

def g1_with_jitter(tau, sigma, sigma_J):
    """Funzione di correlazione g^(1)(tau) con jitter"""
    return np.exp(-tau**2 / (2 * (sigma**2 + sigma_J**2)))

# Parametri della simulazione
sigma = 50e-12   # Larghezza temporale dell'impulso (50 ps)
sigma_J = 20e-12 # Jitter temporale (20 ps)
tau = np.linspace(-300e-12, 300e-12, 1000)  # Intervallo di ritardi (-300 ps a 300 ps)

# Calcoliamo g^(1)(tau) nei due casi
g1_ideal = g1_no_jitter(tau, sigma)
g1_jitter = g1_with_jitter(tau, sigma, sigma_J)

# Plot dei risultati
plt.figure(figsize=(8,5))
plt.plot(tau * 1e12, g1_ideal, label='Senza Jitter', linewidth=2)
plt.plot(tau * 1e12, g1_jitter, label='Con Jitter (σ_J = 20 ps)', linewidth=2, linestyle='dashed')
plt.xlabel('Ritardo τ (ps)')
plt.ylabel('g^(1)(τ)')
plt.legend()
plt.title('Effetto del Jitter sulla Funzione di Correlazione g^(1)(τ)')
plt.grid()
plt.show()