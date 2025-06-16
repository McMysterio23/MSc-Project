import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

def calc_SNR(A, B):
    
    """
    Calculates the signal-to-noise ratio (SNR) in decibels (dB).

    Parameters:
    - A: Signal power or intensity.
    - B: Noise power or intensity.

    Returns:
    - SNR value in dB: 10 * log10(A / B)
    """
    
    return 10 * np.log10(A / B)

def calc_SNR_e(A, B, A_e, B_e):
    
    """
    Calculates the propagated error in the signal-to-noise ratio (SNR).

    Parameters:
    - A: Signal power or intensity.
    - B: Noise power or intensity.
    - A_e: Uncertainty (error) in A.
    - B_e: Uncertainty (error) in B.

    Returns:
    - Propagated uncertainty in SNR (in dB) using error propagation formulas.
    """
    
    return np.sqrt(
    (10*A_e/(A*np.log(10)))**2
    + (10*B_e/(B*np.log(10)))**2
    )

def gaussian(x, A, b, hwhm, d):
    
    """
   Evaluates a Gaussian function.

   Parameters:
   - x: Input array or value.
   - A: Amplitude of the Gaussian.
   - b: Baseline offset.
   - hwhm: Half-width at half-maximum (controls the spread).
   - d: Center of the peak.

   Returns:
   - Value(s) of the Gaussian function at x.
   """
    
    sigma = hwhm * 2 / (2*np.sqrt(2*np.log(2)))
    return A*np.exp(-0.5*((x-d)/sigma)**2) + b


def lorentzian(x, A, b, hwhm, a):
    
    """
   Evaluates a Lorentzian function.

   Parameters:
   - x: Input array or value.
   - A: Amplitude of the Lorentzian.
   - b: Baseline offset.
   - hwhm: Half-width at half-maximum (defines peak width).
   - a: Center position of the peak.

   Returns:
   - Value(s) of the Lorentzian function at x.
   """
    
    # peak = np.exp(-(x-a)**2/w**2)
    peak = (1+(x-a)**2/hwhm**2)**(-1)
    return np.abs(A)*peak + np.abs(b)

def decluster_indices(array, idxs, distance=5):
    """Takes an array and indices of the array. 
    The function returns the index where the array is maximum for groups of indices that are close together.
    
    FELLOW FRIEND of mine description : 
    
    De-clusters close indices by selecting the index of the local maximum within each cluster.

    Parameters:
    - array: Original data array.
    - idxs: Array of candidate indices (e.g., from thresholding).
    - distance: Minimum separation between peaks to consider them distinct.

    Returns:
    - Array of de-clustered peak indices, where each index corresponds to the maximum value in its group.
    
    
    """


    # Then find clusters (because many values are lower than the threshold)
    group_idx = np.argwhere(np.diff(idxs) > distance).flatten()
    # Because we select the elements that split the group, also split the last group from the end of the array.
    group_idx = np.hstack([group_idx, [idxs.size-1]])
    
    # In this loop, the minimum values of arr are used to replace the clusters
    peaks_idxs = []
    temp_idx = 0
    for i in range(len(group_idx)):
        # Start and stop indices of the cluster
        cluster_start_idx = temp_idx
        cluster_end_idx = group_idx[i]
        # Update temp idx
        if i == len(group_idx)-1:
            pass
        else:
            temp_idx = group_idx[i]+1 # Add one because we already used this idx
        
        # Create mask of cluster
        # We need to fill in missing values from the clusters!
        mask = np.clip(np.arange(idxs[cluster_start_idx], idxs[cluster_end_idx]+1, 1), 0, array.size-1)

        # Find minimum arg in cluster mask and add start idx (returned idx is relative to mask)
        peak_idx = np.argmax(array[mask])+idxs[cluster_start_idx]

        
        # Because we took np.diff we are off by 1 idx, correct for this!
        peaks_idxs.append(peak_idx)

    return np.array(peaks_idxs)

def peakfinder(arr, thresh=None, distance_=1, plot=False):
    
    """
   Identifies peaks in a 1D array based on curvature (second derivative).

   Parameters:
   - arr: Input array to search for peaks.
   - thresh: Threshold multiplier for peak sensitivity (default: 2).
   - distance_: Minimum spacing between detected peaks.
   - plot: If True, plots intermediate steps for visualization.

   Returns:
   - Array of peak indices where the most significant changes (dips) in curvature occur.
   """
    
    ddarr = np.diff(np.diff(arr))
    if thresh is None:
        thresh = 2

    # Find an appropriate threshold
    std = np.std(ddarr, ddof=1)
    thresh= -std * thresh

    # Find the dips
    below_threshold = np.argwhere(ddarr < thresh).flatten()

    if plot:
        plt.plot(ddarr/ddarr.max())
        plt.plot(below_threshold, ddarr[below_threshold]/ddarr.max(), '.')
        # plt.show()

    # We should add one, because the greatest negative change in the second derivative happens the index before the peak!
    peak_idx = decluster_indices(arr, below_threshold+1, distance=distance_)

    if plot:
        plt.plot(arr/arr.max())
        plt.plot(peak_idx, arr[peak_idx]/arr.max(), '.')
        plt.show()

    return peak_idx

def find_periodicity(geetwo, plot=False, threshold=None):
    """This is advanced period finding
    First a Fourier transform is used to find an estimate of the period. This estimate is then refined by finding the peak of a
    correlation within the produced uncertainty window.
    
    FELLOW COLLAGUE DESCRIPTION
    
   Determines the period of a periodic signal using a hybrid Fourier-correlation approach.

   Parameters:
   - geetwo: Input 1D array representing a periodic signal.
   - plot: If True, plots the correlation and peak for visual confirmation.
   - threshold: Optional threshold parameter passed to peak detection.

   Returns:
   - period: Estimated period (in samples) between repeating features in the signal.

   Method:
   1. A coarse period estimate is obtained using a Fourier transform.
   2. The estimate is refined by locating the peak of an autocorrelation slice around the estimated region.
    
    
    """
    guess_period, delta_period = find_periodicity_(geetwo, plot=plot, method="fourier", period_error=True, threshold=threshold)
    
    geetwo_preprocessed = geetwo-geetwo.mean()
    # Correlate the signal
    corr = correlate(geetwo_preprocessed, geetwo_preprocessed, mode="same")**2
    middle_idx = np.ceil(corr.size / 2).astype(np.int_)

    lower_bound_idx = np.clip(middle_idx+guess_period-delta_period, middle_idx+int(1/10*delta_period), None)
    upper_bound_idx = np.clip(middle_idx+guess_period+delta_period, lower_bound_idx+1, corr.size)

    corr_slice = corr[lower_bound_idx:upper_bound_idx]

    
    # # Correct for slope to make it easier, removes the effect of the rectangular window.
    # mean_dcorr = np.mean(np.diff(corr[middle_idx:]))
    # corr_slice -= mean_dcorr * np.arange(corr_slice.size)

    # Find peaks
    pk_idx = np.argmax(corr_slice)
    period = pk_idx + lower_bound_idx - middle_idx
    # print("Period:", period)
    

    if plot:
        pk_idx += lower_bound_idx #  This is for the plot
        plt.plot(corr)
        plt.plot(pk_idx, corr[pk_idx], '.')
        plt.show()
    
    return period


def find_periodicity_(geetwo, plot=False, threshold=None, method="fourier", period_error=False):
    """Given an array geetwo of a periodic peaks, return the distance of the peaks in samples, idx_delta.
    
    Fellow Collague description
    
    Computes the periodicity of a signal using either Fourier transform or autocorrelation.

    Parameters:
    - geetwo: Input 1D array with repeating peaks.
    - plot: If True, plots the intermediate steps for verification.
    - threshold: Optional threshold for peak detection sensitivity.
    - method: Either "fourier" (default) or "correlation".
    - period_error: If True (and using "fourier"), also returns the estimated uncertainty.

    Returns:
    - period: Estimated period in number of samples.
    - (optional) delta_period: Estimated error in the period (only with method="fourier" and period_error=True).

    Notes:
    - "fourier" method finds dominant frequency via power spectrum and inverts it to get the period.
    - "correlation" method finds lag between strongest peaks in the autocorrelation of the signal.
    
    
    
    
    
    """

    if method == "fourier":
        padded_geetwo = geetwo #* np.hamming(geetwo.size)
        # padded_geetwo = np.pad(padded_geetwo, padded_geetwo.size*0)
        signal_ft = np.abs(np.fft.rfft(padded_geetwo))**2 # Power spectrum
        signal_freq = np.fft.rfftfreq(padded_geetwo.size)

        # Add the number of idx we do not enter into peakfinder!
        lowpass_cut_freq = 1/geetwo.size

        pks = peakfinder(np.abs(signal_ft)[signal_freq >= lowpass_cut_freq], plot=plot, thresh=threshold) + np.sum(signal_freq < lowpass_cut_freq)
        # pks = pks[signal_freq[pks] > lowpass_cut_freq]
        pk_idx = pks[signal_freq[pks].argmin()]
        pk_freq = signal_freq[pk_idx]

        if plot:
            plt.plot(signal_freq, np.abs(signal_ft[:]/signal_ft.max()))
            plt.plot(signal_freq[pks], np.abs(signal_ft[pks]/signal_ft.max()), ".")
            plt.plot(pk_freq, np.abs(signal_ft[pk_idx])/signal_ft.max(), '.')
            # plt.plot(corr_freq[:], corr_ft[:]/corr_ft.max())
            # plt.plot(geetwo)
            plt.xlim(lowpass_cut_freq, 50*lowpass_cut_freq)
            plt.ylim(0, signal_ft[signal_freq > lowpass_cut_freq].max()/signal_ft.max())
            plt.title("signal ft")
            plt.show()

        # pks = pks[signal_freq[pks] > 0]

        period = 1/pk_freq
        period = np.round(period).astype(np.int_)

        if period_error:
            return period, np.ceil(np.abs(period - 1/signal_freq[pk_idx-1])).astype(np.int_)

    elif method == "correlation":
        # Correlate the signal
        corr = correlate(geetwo, geetwo, mode="same")
        # Find peaks
        pks = peakfinder(corr, thresh=threshold, plot=plot)#find_peaks(-ddcorr, threshold=threshold, distance=distance, prominence=prominance)[0]

        # For visual verification that it is correct!
        if plot:
            plt.plot(corr)
            plt.plot(pks, corr[pks], '.')

        # If there is no periodicity there is no period :)
        if len(pks) <= 1:
            return 0
        elif len(pks) == 2:
            return np.abs(pks[1] - pks[0])
        
        # Find the idx of the highest peak in pks
        highest_peak_pks_idx = corr[pks].argmax()

        # Idx of the correlated signal instead of the pks array
        highest_peak_idx = pks[highest_peak_pks_idx]
        # Find the idx of the peak right of the highest peak
        next_highest_peak_idx = pks[highest_peak_pks_idx+1]

        # For visual verification that it is correct!
        if plot:
            plt.plot(highest_peak_idx, corr[highest_peak_idx], 'o')
            plt.plot(next_highest_peak_idx, corr[next_highest_peak_idx], 'o')
            plt.show()

        period = next_highest_peak_idx - highest_peak_idx
    
    else:
        raise NotImplementedError("Only period finding methods implemented are 'fourier' and 'correlation'.")

    return period

if __name__ == "__main__":
    pass
