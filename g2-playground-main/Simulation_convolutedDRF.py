# -*- coding: utf-8 -*-
"""
Created on Thu May 22 12:30:19 2025

@author: Maccarinelli
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar
import numpy.typing as npt
from pathlib import Path
# from matplotlib.ticker import EngFormatter
# from src.Librerie import Do_Gauss_Fit_v4, gaussian, lorentzian, Do_Lorentzian_Fit, Do_Gauss_Fit_v5, sech2, Do_Sech2_Fit


from src.HBT_analysis import process_geetwo, calculate_photon_ratio_error, lorentzian, make_fit
from src.photon_generator import LightGenerator, QuantumDot, Detector, BeamSplitter, DeadTime, multi_stream_wrapper
from src.plot_utils import plotstream, arrival_distribution, statistics_test
from src.HanburyBrownTwiss import g2_experiment, g2_tdc
from src.utils import gaussian

from src.AlchemistLibrary import get_coincidence_counts_from_stream



# %% Initialization of the main parts of the simulation !

VIEW_BEFORE = False
VIEW_LATER = False


# Specify the brightness, here we relate it to power and wavelength.
power = 0.65*1e-11 # W
wavelength = 935e-9 # nm
brightness = power * wavelength / (6.6e-34 * 3e8)

timing_jitter_detector = 0.150  # ns
timing_jitter_card = 0.025     # ns
timing_jitter = np.sqrt(timing_jitter_card**2+timing_jitter_detector**2)

lg = LightGenerator(
    stream_length=1e-2, 
    pulse_width=0.05, 
    period=20, 
    dark_counts=400, 
    brightness=brightness, 
    extinction_ratio=900, 
    pulse_shape="gaussian")


# lg()  # generate the photon streams
streams = lg.generate()  # call the method to get the streams
emission_Streams = streams

# print(streams)


# initial_emission_streams = [s.copy() for s in lg._streams]  # Store initial emission data


det = Detector(t_jitter=timing_jitter, detector_efficiency=0.945)
bs = BeamSplitter()
deadtime = DeadTime(t_dead=50)

# Plot the photon stream before we apply loss

# statistics_test(lg)

if VIEW_BEFORE :
    statistics_test(lg)
    plt.figure()
    plotstream(lg)
    plt.show()
    
    
    plt.figure()
    arrival_distribution(lg)
    plt.show()

print("-"*10, "Loss from here on", "-"*10, '\n'*3)

# We do HBT
# lg.apply(bs)
split_streams = bs(streams)



for i, stream in enumerate(split_streams):
    print(f"Stream {i} length:", len(stream))
    print(f"Stream {i} sample data:", stream[:10])  # first 10 photons

# arrival_distribution(lg)

print('\n'*3)

# A full detector has deadtime and finite efficiency and jitter.
@multi_stream_wrapper
def full_detector(s):
    return det(deadtime(s))

# Simulate the detector effects: deadtime and finite efficiency.
lg.apply(full_detector)

Streams_Detectors = []
for i, stream in enumerate(split_streams):
    Streams_Detectors.append(full_detector(stream))

# print(len(Streams_Detectors))

# Plot the photon stream after we apply loss
if VIEW_LATER:
    plt.figure()
    statistics_test(lg)
    plotstream(lg)
    plt.show()

# print('\n\n',type(lg), '\n\n', dir(lg))



# %% Search for coincidences in the Good 'ol way


# get_coincidence_counts_from_stream()


STEP_ps = 1
MaxTauDistance = 800


stream1 = Streams_Detectors[0][:]
stream2 = Streams_Detectors[1][:]

coincidence_counts, taus = get_coincidence_counts_from_stream(stream1, stream2,  max_difference=MaxTauDistance, chunk_size=4500, step_size=STEP_ps)


plt.figure(figsize = (8,6))
plt.scatter(taus, coincidence_counts, s = 2, color = 'red')
plt.plot(taus, coincidence_counts)
plt.show()


















# # For the g2 measurement: Total integration time, that needs to repeat the lightstream some number of times.
# # You can't pick lg.stream_length too long because the arrays get huge (and the autocorrelation)!
# int_time = 0.01
# runs = int(int_time/(lg.stream_length*1e-9))

# # Arrays for g2 data
# g2s = []
# e_g2s = []

# with tqdm(total=runs) as pbar:
#     for i in range(runs):
#         tau, g2, e_g2 = g2_tdc(*lg(), delta_tdc=0.1, max_tau=90, method="direct")  # Additionally you can specify 'convolution', but this is much faster!
#         g2s.append(g2)
#         e_g2s.append(e_g2)
#         pbar.update(1)

# # convert data to nparrays
# g2s = np.array(g2s)
# e_g2s = np.array(e_g2s)

# g2 = np.sum(g2s, axis=0)
# e_g2_arr = np.sqrt(np.sum(e_g2s**2, axis=0))

# # Plot g2
# plt.errorbar(tau, g2, fmt="-")#, yerr=e_g2s[i])
# plt.ylim(0)
# plt.xlabel("$\\tau$ [ns]")
# plt.ylabel("g$^2(\\tau)$")
# plt.tight_layout()
# plt.show()

# # Extract again the photon ratio from the simulated g2. It should be close to the value we put in! For square pulses: N_p/N_b = duty cycle * extinction ratio
# b, b_err, geetwo, geetwo_e, period, N_peaks = process_geetwo(g2, tau, plot_fit=True)
# per, per_e = calculate_photon_ratio_error(b, b_err, geetwo, geetwo_e, period, N_peaks)
# print("Extracted Photon Ratio: ",per, "+/-", per_e, "\nTheoretical Photon Ratio: ", lg.duty_cycle*lg.extinction_ratio)