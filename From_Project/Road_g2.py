# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 19:20:53 2025

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
from src.Librerie import Do_Gauss_Fit, gaussian, peakfinder22, decluster_indices
from src.utils import peakfinder


# from src.HBT_analysis import process_geetwo, calculate_photon_ratio_error, lorentzian, make_fit
# from src.photon_generator import LightGenerator, QuantumDot, Detector, BeamSplitter, DeadTime, multi_stream_wrapper, Delay
# from src.plot_utils import plotstream, arrival_distribution, statistics_test
# from src.HanburyBrownTwiss import g2_experiment, g2_td




PATH = Path("RAW")
filename = PATH / 'EOM_-40ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T12_04_38.bin'

data = np.fromfile(filename, dtype=np.uint64)
data = data.reshape(-1, 2)
pippo = np.arange(0,12000,12)   #FOR ANDREA : This is the array of the right sided locations of the bins
# plt.figure()
counts, _ = np.histogram(data[:,0], bins=pippo)     #Creates the Histogram by subdividing the timestamps on the bins already created 
# plt.hist(data[:,0], bins=1000, range=(0,12000))
# plt.xlim([4600,5100])
# plt.show()





exitParams = Do_Gauss_Fit(pippo, counts, True)      #Given the two main object defining the Histogram does a gaussian fit printing out a dataframe of the parameters

bin_centers = (pippo[:-1] + pippo[1:]) / 2          # Array of the positions on the x axis of the centers of the bins
threshold = 0.8 * bin_centers.max()
mask = bin_centers > threshold
print(f"Selected {mask.sum()} bins for baseline averaging.")





def gaussian2(x, A, mu, sigma, d):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + d



# Parametri da 'exitParams'
center = exitParams.loc[3, "Value"]  # Centro del picco
fwhm = exitParams.loc[2, "Value"]    # FWHM
offset_b = exitParams.loc[1, "Value"]  # Offset (se necessario)
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Calcolo della deviazione standard

# Maschere per le diverse regioni
mask_peak = (bin_centers > center - 6 * sigma) & (bin_centers < center + 6 * sigma)
mask_flat = (bin_centers < center - 6 * sigma) | (bin_centers > center + 6 * sigma)
mask_mid = (~mask_peak) & (~mask_flat)

# Calcolo dei fattori di normalizzazione
norm_flat = np.mean(counts[mask_flat])  # Media dei valori nella zona piatta
norm_peak = np.mean(counts[mask_peak])  # Media dei valori nel picco

# Verifica che nessuno dei due sia zero per evitare errori
if norm_flat == 0 or norm_peak == 0:
    raise ValueError("Errore nella normalizzazione: uno dei normalizzatori è zero.")

# Normalizzazione
g2_tau = np.zeros_like(counts, dtype=float)

# Normalizzazione per la zona piatta (valori lontani dal picco)
g2_tau[mask_flat] = counts[mask_flat] / norm_flat  

# Normalizzazione per la zona del picco (valori intorno al centro)
g2_tau[mask_peak] = counts[mask_peak] / norm_peak * np.mean(g2_tau[mask_flat])  # Scala rispetto alla zona piatta

# Zona intermedia tra picco e baseline
g2_tau[mask_mid] = counts[mask_mid] / norm_flat  # Trattata come zona piatta





plt.plot(bin_centers, g2_tau)
plt.axhline(1, linestyle='--', color='gray', label='g₂ baseline')
plt.axvline(center, linestyle='--', color='red', label='Fit center')
plt.title("g₂(τ) Normalized")
plt.xlabel("τ [ps]")
plt.ylabel("g₂(τ)")
plt.xlim([4250,5400])
plt.legend()
plt.show()



