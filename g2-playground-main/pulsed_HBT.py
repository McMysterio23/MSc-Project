import numpy as np
import os, sys
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("P:\My Documents\GitHub\EOM\EOM")
from src.HBT_analysis import process_geetwo, calculate_photon_ratio_error, lorentzian, make_fit
from src.photon_generator import LightGenerator, QuantumDot, Detector, BeamSplitter, DeadTime, multi_stream_wrapper, Delay
from src.plot_utils import plotstream, arrival_distribution, statistics_test
from src.HanburyBrownTwiss import g2_experiment, g2_tdc
from src.utils import gaussian

if __name__ == "__main__":
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
        extinction_ratio=900)
    qd = QuantumDot(lifetime=0.3, interaction_probability=1, extinction_probability=1-1/40)
    det = Detector(t_jitter=timing_jitter, detector_efficiency=1.)
    bs = BeamSplitter(photon_length=qd.lifetime, indistinguishability=1.)
    deadtime = DeadTime(t_dead=50)
    delay = Delay(delay=lg.period)

    # A full detector has deadtime and finite efficiency and jitter.
    @multi_stream_wrapper
    def full_detector(s):
        return det(deadtime(s))

    # Apply the QD and see the central peak disappear!
    #lg.apply(qd)
    lg.apply(bs)
    lg.apply(full_detector)

    # For the g2 measurement: Total integration time, that needs to repeat the lightstream some number of times.
    # You can't pick lg.stream_length too long because the arrays get huge (and the autocorrelation)!
    int_time = 0.01
    runs = int(int_time/(lg.stream_length*1e-9))

    # Get g2 data for realistic detector
    g2s = []
    e_g2s = []

    with tqdm(total=runs) as pbar:
        for i in range(runs):
            tau, g2, e_g2 = g2_tdc(*lg(), delta_tdc=0.1, max_tau=100, method="direct")
            g2s.append(g2)
            e_g2s.append(e_g2)
            pbar.update(1)

    # convert data to nparrays
    g2s = np.array(g2s)
    e_g2s = np.array(e_g2s)

    g2 = np.sum(g2s, axis=0)
    e_g2_arr = np.sqrt(np.sum(e_g2s**2, axis=0))

    # Plot g2
    plt.errorbar(tau, g2, fmt="-")#, yerr=e_g2s[i])
    #plt.scatter(tau, g2, s=2)
    plt.ylim(0)
    plt.xlabel("$\\tau$ [ns]")
    plt.ylabel("g$^2(\\tau)$")
    plt.tight_layout()
    plt.show()
