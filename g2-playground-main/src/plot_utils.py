import matplotlib.pyplot as plt
import numpy as np

def plotstream(lg, divider=None):
    
    """
    Plots a histogram of photon arrival times modulo the laser period.

    Parameters:
    - lg: A callable that returns a stream of photon arrival times (as a list or array).
          This object is expected to have attributes:
              - period: the laser repetition period
              - pulse_width: the laser pulse width
    - divider: Optional horizontal line value to indicate a reference/count threshold.

    Behavior:
    - If multiple streams are returned (as a list), each stream is processed individually.
    - For each stream:
        - Times are wrapped modulo the laser period.
        - A histogram is generated with bin width = pulse_width / 5.
        - The histogram is plotted on a semi-log scale (log y-axis).
        - If 'divider' is provided, a horizontal line is drawn for reference.
    - Displays the plot.
    """
    
    stream = lg()
    if type(stream) is list:
        #stream = np.hstack([s for s in stream])
        streams = stream
        for stream in streams:
            times = np.mod(stream, lg.period)
            bins = np.arange(0, lg.period, lg.pulse_width/5)
            counts, bins = np.histogram(times, bins=bins)
            plt.plot(bins[:-1], counts, '.')
            plt.plot([bins[0], bins[-2]], [divider,divider])
    else:
        times = np.mod(stream, lg.period)
        bins = np.arange(0, lg.period, lg.pulse_width/5)
        counts, bins = np.histogram(times, bins=bins)
        plt.plot(bins[:-1], counts, '.')
        plt.plot([bins[0], bins[-2]], [divider,divider])
    plt.yscale('log')
    # plt.xscale("log")
    plt.xlabel("Time $t$ [ns]")
    plt.ylabel("Counts [-]")
    plt.title("Histogram of photon stream mod period")
    plt.show()

def arrival_distribution(lg):
    
    """
    Plots a histogram of time differences between consecutive photon arrivals.

    Parameters:
    - lg: A callable that returns a stream of photon arrival times (as a list or array).
          This object is expected to have attributes:
              - period: the laser repetition period
              - pulse_width: the laser pulse width

    Behavior:
    - If multiple streams are returned (as a list), each stream is processed individually.
    - For each stream:
        - Calculates time differences between consecutive photon events (delta t).
        - Plots a histogram of these differences with bin width = pulse_width / 5.
    - The y-axis is logarithmic to better visualize a wide range of counts.
    - Displays the plot with appropriate labels and title.
    """
    
    stream = lg()
    if type(stream) is list:
        #stream = np.hstack([s for s in stream])
        streams = stream
        for stream in streams:
            dstream = np.diff(stream)
            plt.hist(dstream, bins=np.arange(0, lg.period, lg.pulse_width/5))

    else:
        dstream = np.diff(stream)
        plt.hist(dstream, bins=np.arange(0, lg.period, lg.pulse_width/5))
    plt.yscale('log')
    plt.xlabel("Time difference $\\tau$ [ns]")
    plt.ylabel("Counts [-]")
    plt.title("Histogram of photon stream arrival time difference")
    plt.show()

def statistics_test(lg):
    
    """
   Performs statistical analysis on a photon stream to estimate signal and background contributions.

   Parameters:
   - lg: A callable returning a photon arrival time stream (list or array), with attributes:
         - period: laser repetition period
         - pulse_width: laser pulse width
         - N_pulses: total number of laser pulses
         - mean_photons_in_pulse: expected photons per pulse
         - mean_photons_in_background: expected background photons per pulse
         - mean_darkcounts: expected dark counts per pulse

   Behavior:
   - Flattens the stream if it contains multiple entries.
   - Calculates a histogram of photon arrival times modulo the laser period.
   - Prints three methods to estimate signal (pulse) vs. background photons:
       1. **Time dividing**: Counts photons inside vs. outside the pulse window.
       2. **Integration method**: Uses a flat background estimate to isolate pulse signal.
       3. **Direct counting**: Identifies counts above a noise threshold (~4σ above mean).
   - Each method outputs:
       - Estimated number of signal and background photons per pulse.
       - Photon-to-Background Ratio (PER).
   
   Notes:
   - This is useful to evaluate the efficiency of time-gated photon detection.
   - Optional: Uncomment `plotstream(lg, divider)` to visualize the stream with the threshold.
   """
    
    stream = lg()
    if type(stream) is list:
        stream = np.hstack([s for s in stream])

    times = np.mod(stream, lg.period)
    bins = np.arange(0, lg.period, lg.pulse_width/5)
    counts, bins = np.histogram(times, bins=bins)

    print("Theory")
    print("N_pulse: ", lg.mean_photons_in_pulse*lg.N_pulses, "\t N_bg: ", lg.mean_photons_in_background*lg.N_pulses, "\t PER: ", lg.mean_photons_in_pulse/lg.mean_photons_in_background)

    print("Time dividing")
    N_pulse = np.sum(counts[bins[:-1]<=lg.pulse_width])
    N_background = np.sum(counts[bins[:-1]>lg.pulse_width])
    print("N_pulse: ", N_pulse/lg.N_pulses, "\t N_bg: ", N_background/lg.N_pulses, "\t PER: ", N_pulse/N_background)
    
    divider = (lg.mean_photons_in_background + lg.mean_darkcounts)*lg.N_pulses / counts.size #counts.mean()

    print("Integration")
    # Method 1: integration
    N_background = divider * bins.size 
    N_pulse = np.sum(counts) - N_background
    print(" N_pulse: ", N_pulse/lg.N_pulses, "\t N_bg: ", N_background/lg.N_pulses, "\t PER: ", N_pulse/N_background)

    # plotstream(lg, divider)

    print("Direct counting")
    # Method 2: direct counting
    divider += 4*np.sqrt(divider) # add 4 std -> 99.9% is within this bound
    N_pulse = np.sum(counts[counts > divider]-divider)
    N_background = np.sum(counts[counts < divider]-divider) + divider*counts.size
    print(" N_pulse: ", N_pulse/lg.N_pulses, "\t N_bg: ", N_background/lg.N_pulses, "\t PER: ", N_pulse/N_background)
