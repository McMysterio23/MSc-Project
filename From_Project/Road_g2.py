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
from src.Librerie import Do_Gauss_Fit, gaussian


# from src.HBT_analysis import process_geetwo, calculate_photon_ratio_error, lorentzian, make_fit
# from src.photon_generator import LightGenerator, QuantumDot, Detector, BeamSplitter, DeadTime, multi_stream_wrapper, Delay
# from src.plot_utils import plotstream, arrival_distribution, statistics_test
# from src.HanburyBrownTwiss import g2_experiment, g2_td

PATH = Path("RAW")
filename = PATH / 'EOM_-40ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T12_04_38.bin'

data = np.fromfile(filename, dtype=np.uint64)
data = data.reshape(-1, 2)
pippo = np.arange(0,12000,12)
# plt.figure()
counts, _ = np.histogram(data[:,0], bins=pippo)
# plt.hist(data[:,0], bins=1000, range=(0,12000))
# plt.xlim([4600,5100])
# plt.show()





exitParams = Do_Gauss_Fit(pippo, counts, True)

bin_centers = (pippo[:-1] + pippo[1:]) / 2
threshold = 0.8 * bin_centers.max()
mask = bin_centers > threshold
print(f"Selected {mask.sum()} bins for baseline averaging.")
baseline = np.mean(counts[mask])
g2_tau = counts / baseline



plt.scatter(bin_centers, g2_tau, label='g₂(τ)')
plt.xlabel('τ [ps]')
plt.ylabel('g₂(τ)')
plt.title('Normalized Second-Order Correlation')
plt.grid(True)
plt.legend()
plt.show()

