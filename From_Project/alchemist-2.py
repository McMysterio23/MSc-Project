"""
Created on Mon Apr 28 14:26:36 2025

@author: Maccarinelli
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar
import numpy.typing as npt
from pathlib import Path
from matplotlib.ticker import EngFormatter
from src.Librerie import Do_Gauss_Fit_v4, gaussian, lorentzian, Do_Lorentzian_Fit, Do_Gauss_Fit_v5, sech2, Do_Sech2_Fit



# %% 1st part of the script !!!!!!!!!!!
PATH = Path("RAW")
folder_path = PATH
file_list = [f.name for f in folder_path.iterdir() if f.is_file() and f.suffix == '.bin']
names = [file_list[0], file_list[1], file_list[2], file_list[3], file_list[4], file_list[5], file_list[6], file_list[7],
         file_list[8], file_list[9]]

PATH_TSUNAMI = Path('ModeLockedLaser/NewBeginning')
folder_path_tsu = PATH_TSUNAMI
file_list_TSUNAMI_bin = [f.name for f in folder_path_tsu.iterdir() if f.is_file() and f.suffix == '.bin']
names_tsu = [file_list_TSUNAMI_bin[0], file_list_TSUNAMI_bin[1], file_list_TSUNAMI_bin[2]]

"""
At the moment there is in the data folder this set of files:
    
    EOM_-10v2ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-04-23T15_23_17.bin


    EOM_-10v2ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-04-23T15_23_17.bin


    EOM_-25ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-04-23T15_38_45.bin


    EOM_-25ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-04-23T15_38_45.bin


    EOM_-35ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-04-23T15_47_05.bin


    EOM_-35ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-04-23T15_47_05.bin


    EOM_-50ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-04-23T16_12_47.bin


    EOM_-50ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-04-23T16_12_47.bin


    EOM_-60ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-04-23T16_33_20.bin


    EOM_-60ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-04-23T16_33_20.bin
"""



fname_detector2 = PATH / names[0]
fname_detector3 = PATH / names[1]

@njit
def sparse_difference_matrix(stream1, stream2, max_difference, j_start=0):
    '''Gets all relevant differences between stream1 and stream2.
    Assumes both stream arrays are sorted!, loop is stopped when distance is greater than max_difference.
    Use same units for all inputs!
    
    Inputs:
        stream1 (nparray): 1D array containing photon emission times at output/detector 1
        stream2 (nparray): 1D array containing photon emission times at output/detector 2
        max_difference (float): time difference at which the differences are truncated

    Ouputs:
        time_differences (nparray): 1D array containing all time differences < max_difference between arrays
    '''
    # Alocate memory for output
    time_differences = np.empty(stream1.size**2, dtype=np.int64)
    # But we will not use the whole array, so keep track of where to end
    time_differences_idx = 0

    # Memory for the inner loop
    temp_differences = np.empty(stream2.size)

    # For the inner loop, the last useful time idx of stream2
    # j_start = 0

    for i in range(len(stream1)):

        for store_index, j in enumerate(range(j_start, len(stream2))):
            # Calc time differnces
            temp_differences[store_index] = stream2[j]-stream1[i]

            # Check if following calculations are useful
            if abs(temp_differences[store_index]) > max_difference: 
                if temp_differences[store_index] < 0:
                    j_start = j+1
                else:
                    break
        # Write to output array
        if time_differences_idx+store_index > time_differences.size:
            print("Time difference overflow")
            break
            raise OverflowError("time_differences is too small, think about it you lazy sob")
        time_differences[time_differences_idx:time_differences_idx+store_index] = temp_differences[:store_index]
        # Update index of output array
        time_differences_idx += store_index
    if time_differences_idx >= time_differences.size:
        raise OverflowError("Trying to read higher idx than exists!")
        # time_differences_idx = time_differences.size

    # Return only the indices we wrote to
    return time_differences[:time_differences_idx], j_start

@njit(fastmath=True, parallel=True, nogil=True)
def sparse_coincidence_counts_jit(stream1:npt.NDArray[np.int64], stream2:npt.NDArray[np.int64], max_difference:int, chunk_size=10000, step_size = 100000, pbar=ProgressBar()):
    pbar.update(1)

    N_bins = int(2*np.ceil(max_difference/step_size)+1)
    N_chunks = int(np.ceil(stream1.size/chunk_size))
    hist = np.zeros(N_bins, dtype=np.int64)
    pbar.update(1)
    j_start=0


    for i in prange(N_chunks):
        chunk_start_idx = i*chunk_size
        chunk_stop_idx = min(stream1.size, chunk_start_idx+chunk_size)
        time_differences, j_start = sparse_difference_matrix(stream1[chunk_start_idx:chunk_stop_idx], stream2, max_difference, j_start)
        hist += np.histogram(time_differences, bins=N_bins, range=(-max_difference-0.5*step_size, max_difference+0.5*step_size))[0]
        pbar.update(1)
    
    return hist

def get_coincidence_counts_from_stream(stream1, stream2, max_difference, chunk_size, step_size):
    with ProgressBar(total=int(np.ceil(stream1.size/chunk_size))+2) as pbar:
        coincidence_counts = sparse_coincidence_counts_jit(stream1.astype(np.int64), stream2.astype(np.int64), max_difference, chunk_size, step_size, pbar)
    bins=np.arange(-max_difference, max_difference+step_size, step_size)
    
    return coincidence_counts, bins



def get_coincidence_counts_from_files(
        fname1,
        fname2,
        startstop_clock_ps = 15200,
        maxtime_ps =  int(0.001e12), # ps,
        stepsize_ps = 20000,
        chunk_size = 10000, # no. hits in a chunk
        ):
    tags1 = np.fromfile(fname1, dtype=np.uint64, sep="")
    tags2 = np.fromfile(fname2, dtype=np.uint64, sep="")
    shaped_tags1 = tags1.reshape((-1, 2))
    shaped_tags2 = tags2.reshape((-1, 2))
    shaped_tags1[::,1] *= startstop_clock_ps
    shaped_tags2[::,1] *= startstop_clock_ps
    timetags_ps1 = shaped_tags1.sum(1)
    timetags_ps2 = shaped_tags2.sum(1)
    del shaped_tags1, shaped_tags2
    return get_coincidence_counts_from_stream(timetags_ps1, timetags_ps2, maxtime_ps, chunk_size=chunk_size, step_size=stepsize_ps)





def normalize(filename1, filename2, Counts_Coincident, acq_Time, startstop_clock_ps=15200, step_size_ps=1, min_expected_counts=5):
    """
    Normalizes the raw coincidence counts into g2(τ).
    Includes protection against very small bin sizes.
    """
    # Read the time tags from the files
    tags1 = np.fromfile(filename1, dtype=np.uint64, sep="")
    tags2 = np.fromfile(filename2, dtype=np.uint64, sep="")
    
    # Reshape the tags into pairs (tag, timestamp)
    shaped_tags1 = tags1.reshape((-1, 2))
    shaped_tags2 = tags2.reshape((-1, 2))
    
    # Scale the timestamps
    shaped_tags1[:, 1] *= startstop_clock_ps
    shaped_tags2[:, 1] *= startstop_clock_ps
    
    timetags_ps1 = shaped_tags1[:, 1]
    timetags_ps2 = shaped_tags2[:, 1]
    
    del shaped_tags1, shaped_tags2  # Free memory
    
    # Photon rates [photons per second]
    rate1 = len(timetags_ps1) / acq_Time
    rate2 = len(timetags_ps2) / acq_Time
    
    print(f"Rate of acquisition for Detector 2 is {rate1}\n")
    print(f"Rate of acquisition for Detector 3 is {rate2}\n")

    # Calculate expected random coincidences per bin
    expected_counts_per_bin = rate1 * rate2 * (step_size_ps * 1e-12) * acq_Time
    
    # Debug: print expected counts per bin to verify values
    print(f"Expected counts per bin: {expected_counts_per_bin:.6e}")
    
    # Check if expected counts per bin are too small
    if expected_counts_per_bin < min_expected_counts:
        print(f"Warning: expected counts per bin ({expected_counts_per_bin:.2e}) is too low for reliable normalization.")
        print("Consider using a larger bin size or smoothing the data after normalization.")
    
    # Normalize counts
    normalized_counts = Counts_Coincident / expected_counts_per_bin
    
    #compute the poissonian error related to each of these normalized counts 
    # Error is being computed in the form of shot noise
    err_Ncounts = np.sqrt(Counts_Coincident) / expected_counts_per_bin

    
    return normalized_counts, err_Ncounts


STEP_ps = 4
MaxTauDistance = 3000000

coincidence_counts, taus = get_coincidence_counts_from_files(fname_detector2, fname_detector3, stepsize_ps=STEP_ps, maxtime_ps=MaxTauDistance, chunk_size=15000)

print(f"current binsize : {STEP_ps}\n")

# UNCOMMENT IF YOU DESIRE TO LOOK AT THE HISTOGRAM BEFORE THE NORMALIZATION

# plt.figure()  # <- This line ensures a new figure for each iteration
# plt.scatter(taus, coincidence_counts, s=0.8)
# plt.xlabel(r"$\Delta \tau$ [ps]")
# plt.ylabel("Counts")
# plt.show()


Ncounts, ErrNcounts = normalize(
    fname_detector2, fname_detector3,
    coincidence_counts,
    15,  
    step_size_ps=STEP_ps  
)


# %% MAIN PLOTTING CELL FOR SECOND ORDER CORRELATION FUNCTIONS FROM THE COINCIDENCE HISTOGRAMS !!!!!!!!!!

plt.figure()
plt.title(f"Coincidences over pulses with optical length of 65.1 ps and binsize {STEP_ps}ps")
# plt.plot(taus, Ncounts, linewidth=0.8, color='#FF9B00', label = 'Line')  
# plt.scatter(taus, Ncounts, s=4, color='black', label = 'Data Points')

plt.plot(taus, coincidence_counts, linewidth=0.8, color='#FF9B00', label = 'Line')  
plt.scatter(taus, coincidence_counts, s=4, color='black', label = 'Data Points')


# # Axis labels bigger
# plt.xlabel(r"$\Delta \tau$ [ps]", fontsize=16)
# plt.ylabel(r"$g_{2}(\tau)$", fontsize=16)

# Axis labels bigger
plt.xlabel(r"$\Delta \tau$ [ps]", fontsize=16)
plt.ylabel(r"Coincidence Counts", fontsize=16)

# Increase tick size
plt.xticks(fontsize=10)
plt.yticks(fontsize=13)

# Legend (if you want it)
# plt.legend(fontsize=14, loc='best', frameon=True)
plt.xlim((-73400, +64200))
plt.ylim((-3, +3500))
plt.tight_layout()
plt.show()



plt.figure()
plt.title(f"Second Order Correlation Function for -10ps delay pulses and binsize {STEP_ps}ps")
plt.plot(taus, Ncounts, linewidth=0.8, color='#FF9B00', label='Line')  

# Aggiungi scatter plot con le barre di errore
plt.errorbar(taus, Ncounts, yerr=ErrNcounts, fmt='o', color='black', label='Data Points', markersize=4, elinewidth=1, capsize=3)

# Axis labels bigger
plt.xlabel(r"$\Delta \tau$ [ps]", fontsize=16)
plt.ylabel(r"$g_{2}(\tau)$", fontsize=16)

# Increase tick size
plt.xticks(fontsize=10)
plt.yticks(fontsize=13)





# Legend (if you want it)
plt.legend(fontsize=14, loc='best', frameon=True)
plt.xlim((-150000, +150000))
plt.ylim((-3, +220))
plt.tight_layout()
plt.show()





# %% 2nd PART OF THE SCRIPT : IDENTIFY THE FITTING INTERVALS

from src.Librerie import gaussian, skewed_gaussian, Do_Gauss_Fit, peakfinder22

# If taus are bin centers, convert them to bin edges
step_size = taus[1] - taus[0]  # Assuming uniform spacing
taus_edges = np.concatenate(([taus[0] - step_size / 2], taus + step_size / 2))


print(f"The maximum distance being considered for the generation of the intervals of fit is : {MaxTauDistance}")

def Fitting_Intervals(First_Interval, Pulser_periodicity = 15200, maxtau = MaxTauDistance):
    lim_left_zero, lim_right_zero = First_Interval
    
    ### Positive direction
    Right_Borders = np.arange(lim_right_zero, maxtau, Pulser_periodicity)
    Left_Borders = np.arange(lim_left_zero, maxtau, Pulser_periodicity)
    
    ### Negative direction
    Left_Borders_Opp = np.arange(lim_left_zero - Pulser_periodicity, -maxtau, -Pulser_periodicity)
    Right_Borders_Opp = np.arange(lim_right_zero - Pulser_periodicity, -maxtau, -Pulser_periodicity)
    
    # Reverse the negative direction arrays
    Left_Borders_Opp_reversed = Left_Borders_Opp[::-1]
    Right_Borders_Opp_reversed = Right_Borders_Opp[::-1]
    
    # Concatenate the positive and negative direction intervals
    ALL_LEFT_BORDERS = np.concatenate([Left_Borders_Opp_reversed, Left_Borders])
    ALL_RIGHT_BORDERS = np.concatenate([Right_Borders_Opp_reversed, Right_Borders])
    
    return ALL_LEFT_BORDERS, ALL_RIGHT_BORDERS
    
    
    
I_primo = (10283, 10631)

LEFT, RIGHT = Fitting_Intervals(I_primo)


#VISUAL CHECK !!!

plt.figure()
plt.title(f"Second Order Correlation Function for -10ps delay pulses and binsize {STEP_ps}ps")
plt.plot(taus, Ncounts, linewidth=0.8, color='#FF9B00', label = 'Line')  
plt.scatter(taus, Ncounts, s=4, color='black', label = 'Data Points')

# Axis labels bigger
plt.xlabel(r"$\Delta \tau$ [ps]", fontsize=16)
plt.ylabel(r"$g_{2}(\tau)$", fontsize=16)




for i, j in zip(LEFT, RIGHT):
    plt.axvline(i, color='red', linestyle='--')   # linea tratteggiata per il bordo sinistro
    plt.axvline(j, color='red', linestyle='-')  # linea continua per il bordo destro


# Increase tick size
plt.xticks(fontsize=10)
plt.yticks(fontsize=13)

# Legend (if you want it)
# plt.legend(fontsize=14, loc='best', frameon=True)
# plt.xlim((10250, +55000))
plt.xlim((+2500000, +3100000))
plt.ylim((-3, +150))
plt.tight_layout()
plt.show()

# %% 3RD PART OF THE SCRIPT : FIT FOR EACH INTERVAL, SAVING THE DATAFRAMES & SEARCH FOR MINIMUM

from src.Librerie import Do_Gauss_Fit_v4

results = []
indice = 1
for left, right in zip(LEFT, RIGHT):
    
    
    #Creation of the mask
    selection_mask = (taus > left) & (taus < right)
    
    #Restricting the arrays to the current interval being studied !
    Selection_Taus = taus[selection_mask]
    Selected_Ncounts = Ncounts[selection_mask]
    selected_errors = ErrNcounts[selection_mask]

    # Filter out non valid errors !!! (Avoiding errors that are NaN or Zero)
    valid_mask = (selected_errors > 0) & np.isfinite(selected_errors) & np.isfinite(Selected_Ncounts)
    Selection_Taus = Selection_Taus[valid_mask]
    Selected_Ncounts = Selected_Ncounts[valid_mask]
    selected_errors = selected_errors[valid_mask]


    #DEBUGGING PRINTS
    print(f"\n\n We are currently at the ({indice}-th) fit of this current file !")
    print(f"left: {left}, right: {right}")
    print(f"Selection_Taus shape: {Selection_Taus.shape}")
    print(f"Selected_Ncounts shape: {Selected_Ncounts.shape}")
    print(f"Selected_ErrNcounts shape: {selected_errors.shape}")


    #FINAL CHECK BEFORE CALLING THE FIT FUNCTION
    if Selection_Taus.shape[0] == Selected_Ncounts.shape[0] == selected_errors.shape[0] and Selection_Taus.shape[0] > 0:
        try:
            results.append(Do_Gauss_Fit_v4(Selection_Taus, Selected_Ncounts, selected_errors, True))
        except Exception as e:
            print(f"Warning: fit failed at interval ({indice}) with error: {e}")
    else:
        print("Warning: mismatch or empty selection, skipping this interval")

    indice +=1

centri = []
for i in range(len(results)):
    centri.append(results[i].loc[3]["Value"])
    
FWHMs = []
for i in range(len(results)):
    FWHMs.append(results[i].loc[2]["Value"])
    
errFWHMs = []
for i in range(len(results)):
    errFWHMs.append(results[i].loc[2]["Uncertainty"])

errCentri = []
for i in range(len(results)):
    errCentri.append(results[i].loc[3]["Uncertainty"])

    
centri = np.array(centri)
errCentri = np.array(errCentri)    
ConvertedCentri = centri / 1e6
ConvertederrCentri = errCentri / 1e6

plt.figure(figsize=(10,6))
plt.errorbar(ConvertedCentri, FWHMs, xerr = ConvertederrCentri, yerr=errFWHMs, fmt='o', color='black', label='Data Points', markersize=2, elinewidth=1, capsize=1.5, ecolor='red')
plt.title(r"$g_{2}(\tau)$ Peaks : FWHMs vs Centers ")
plt.xlabel(r'Taus [$\mu s$]')
plt.ylabel('FWHMs [ps]')
plt.tight_layout()
plt.show()


index_min = np.argmin(FWHMs)
print(f"\n\nThe Index number in the array of the FWHMs where the array has its minimun is : {index_min}")
print(f"\nThe peak with the minimum FWHM, which is {FWHMs[index_min]}[ps] is located in {centri[index_min]} [ps]")
        
Offset = -centri[index_min]
print(f"\nThe Offset that will be added to the tau axis is :{Offset} [ps]")

# %% # %% 4th PART OF THE SCRIPT : Add the Offset to all relevant arrays

"""
IMPORTANT !!!!!!!!!!!

THIS CELL HAS TO BE EXECUTED ONLY ONCE, SINCE IT SHIFTS EVERY TIME THE ARRAYS.
FOR THIS REASON YOU MAY WANT TO AVOID RUNNING SEVERAL TIMES THIS OPERATION !!!!
"""




#Correction of the taus array
taus = taus+Offset

#Correction of the boundary arrays
LEFT = LEFT + Offset
RIGHT = RIGHT + Offset

# %% 5th part of the script : Visual check that the intervals' boundaries have correctly shifted !!
plt.figure()
plt.title(f"Second Order Correlation Function for -50ps delay pulses and binsize {STEP_ps}ps")
plt.plot(taus, Ncounts, linewidth=0.8, color='#FF9B00', label='Line')  

# Aggiungi scatter plot con le barre di errore
plt.errorbar(taus, Ncounts, yerr=ErrNcounts, fmt='o', color='black', label='Data Points', markersize=4, elinewidth=1, capsize=3)

# Axis labels bigger
plt.xlabel(r"$\Delta \tau$ [ps]", fontsize=16)
plt.ylabel(r"$g_{2}(\tau)$", fontsize=16)

# Increase tick size
plt.xticks(fontsize=10)
plt.yticks(fontsize=13)


for i, j in zip(LEFT, RIGHT):
    plt.axvline(i, color='red', linestyle='--')   # Dashed red line for left boundaries
    plt.axvline(j, color='red', linestyle='-')  # continuous red line for the right boundaries


# Legend (if you want it)
plt.legend(fontsize=14, loc='best', frameon=True)
plt.xlim((-800, +1000))
plt.ylim((-3, +180))
plt.tight_layout()
plt.show()


# %% Final round of fits, this time in the correctly adjusted arrays after the shifts 
results = []
indice = 1
for left, right in zip(LEFT, RIGHT):
    
    
    #Creation of the mask
    selection_mask = (taus > left) & (taus < right)
    
    #Restricting the arrays to the current interval being studied !
    Selection_Taus = taus[selection_mask]
    Selected_Ncounts = Ncounts[selection_mask]
    selected_errors = ErrNcounts[selection_mask]

    # Filter out non valid errors !!! (Avoiding errors that are NaN or Zero)
    valid_mask = (selected_errors > 0) & np.isfinite(selected_errors) & np.isfinite(Selected_Ncounts)
    Selection_Taus = Selection_Taus[valid_mask]
    Selected_Ncounts = Selected_Ncounts[valid_mask]
    selected_errors = selected_errors[valid_mask]


    #DEBUGGING PRINTS
    print(f"\n\n We are currently at the ({indice}-th) fit of this current file !")
    print(f"left: {left}, right: {right}")
    print(f"Selection_Taus shape: {Selection_Taus.shape}")
    print(f"Selected_Ncounts shape: {Selected_Ncounts.shape}")
    print(f"Selected_ErrNcounts shape: {selected_errors.shape}")


    #FINAL CHECK BEFORE CALLING THE FIT FUNCTION
    if Selection_Taus.shape[0] == Selected_Ncounts.shape[0] == selected_errors.shape[0] and Selection_Taus.shape[0] > 0:
        try:
            results.append(Do_Gauss_Fit_v4(Selection_Taus, Selected_Ncounts, selected_errors, True))
        except Exception as e:
            print(f"Warning: fit failed at interval ({indice}) with error: {e}")
    else:
        print("Warning: mismatch or empty selection, skipping this interval")

    indice +=1
    



# %% Saving the pandas dataframes in a txt file 
import os

# # Create a folder if it doesn't exist
# os.makedirs(PATH/"fit_results_-10psDelay_4psStepSize", exist_ok=True)

output_dir = f"fit_results_-60ps_Delay_{STEP_ps}psStepSize"
os.makedirs(output_dir, exist_ok=True)  # creates the folder if it doesn't exist

# Define the Figures subfolder path
figures_dir = os.path.join(output_dir, "Images")

# Create the Figures subfolder if it doesn't exist
os.makedirs(figures_dir, exist_ok=True)

# Now save
for i, df in enumerate(results):
    df.to_csv(f"{output_dir}/results_{i}.csv", index=False)
    
# %% Analyzing the results obtained

"""

ONLY RUN THIS CELL AFTER PROCEDING TO RUN THE PREVIOUS ONE. OTHERWISE THE DATAFRAME ARRAY "results", WON'T BE READY TO BE OPEN !


#HOW TO ACCESS THE RESULTING DATAFRAMES ?!?!?!?!?!?!?!?!    

FOR A LOOK TO A SINGULAR DATAFRAME USE THE FOLLOWING COMMAND
print(results[0])

TO ACCESS A SPECIFIC ELEMENT OF A SPECIFIC ARRAY YOU HAVE TO KNOW 3 THINGS :
    1 ) THE INDEX OF THE DATAFRAME YOU'RE INTERESTED IN;
    2 ) THE STRING WITH THE NAME OF THE COLUMN YOU'RE INTERESTED IN'
    3 ) THE INDEX OF ROW IN THE COLUMN YOU JUST SELECTED !!!
    
    
THIS IS AN EXAMPLE OF A SINGLE DATAFRAME
  Parameter         Value  Uncertainty
0         A  1.359898e+02     0.201528
1         b  3.301681e-01     0.032404
2      FWHM  7.377636e+01     0.127867
3         d -1.494368e+06     0.053264  

THIS IS THE CALL REQUIRED TO ACCESS TO THE VALUE -1.494368e+06 
results[0].loc["Value"][3]


"""





centri = []
for i in range(len(results)):
    centri.append(results[i].loc[3]["Value"])
    
FWHMs = []
for i in range(len(results)):
    FWHMs.append(results[i].loc[2]["Value"])
    
errFWHMs = []
for i in range(len(results)):
    errFWHMs.append(results[i].loc[2]["Uncertainty"])

errCentri = []
for i in range(len(results)):
    errCentri.append(results[i].loc[3]["Uncertainty"])

    


centri = np.array(centri)
errCentri = np.array(errCentri)    
ConvertedCentri = centri / 1e6
ConvertederrCentri = errCentri / 1e6

plt.figure(figsize=(10,6))
plt.errorbar(ConvertedCentri, FWHMs, xerr = ConvertederrCentri, yerr=errFWHMs, fmt='o', color='black', label='Data Points', markersize=2, elinewidth=1, capsize=1.5, ecolor='red')
    
plt.title(r"$g_{2}(\tau)$ Peaks : FWHMs vs Centers ")
plt.xlabel(r'$\tau$ [$\mu s$]')
plt.ylabel('FWHMs [ps]')
plt.tight_layout()

# Define the filename (standardized name inside the Figures folder)
figure_filename = os.path.join(figures_dir, "FWHM_vs_Taus.png")
plt.savefig(figure_filename, dpi=300)
figure_filename = os.path.join(figures_dir, "FWHM_vs_Taus.pdf")

# Save the figure
plt.savefig(figure_filename, dpi=300)

plt.show()


# %% The Research of the offset in the Tau axis to identify the actual \tau = 0 peak !!!!

centers = []
for i in range(len(results)):
    # value = results[i].loc[3]["Value"]
    # print(f"i={i}, type={type(value)}, value={value}")
    centers.append(results[i].loc[3]["Value"])

errCenters= []
for i in range(len(results)):
    errCenters.append(results[i].loc[3]["Uncertainty"])

npcenters = np.array(centers)
nperrors = np.array(errCenters)

# Compute repetition rates & errors
repetition_rates = npcenters[1:] - npcenters[:-1]
repetition_errors = np.sqrt(nperrors[1:]**2 + nperrors[:-1]**2)

# Compute weighted mean and uncertainty
weights = 1 / repetition_errors**2
weighted_mean = np.sum(repetition_rates * weights) / np.sum(weights)
weighted_error = 1 / np.sqrt(np.sum(weights))

print("Weighted repetition rate:", weighted_mean)
print("Uncertainty:", weighted_error)

# %% WL's requested plot !


npConvcenters = np.array(centers) / 1e6
npConverrors = np.array(errCenters) / 1e6

errCenters= []
for i in range(len(results)):
    errCenters.append(results[i].loc[3]["Uncertainty"])

npConverrors = np.array(errCenters) / 1e6

# Midpoints (x-axis)
midpoints = (npConvcenters[1:] + npConvcenters[:-1]) / 2

# Differences (y-axis)
repetition_rates = npConvcenters[1:] - npConvcenters[:-1]
repetition_errors = np.sqrt(npConverrors[1:]**2 + npConverrors[:-1]**2)

plt.figure(figsize=(8, 5))
plt.errorbar(npConvcenters[:-1], repetition_rates, yerr=repetition_errors, color="black", fmt='.', ecolor='red', capsize=3)
plt.xlabel(r'$\tau$ $(\mu s)$ [first center of each pair]')
plt.ylabel(r'$\Delta \tau$ (ps) [difference between centers]')
plt.title('Pulse Spacing vs τ (anchored to first center)')
plt.grid(True)
plt.tight_layout()

# Define the filename (standardized name inside the Figures folder)
figure_filename = os.path.join(figures_dir, "PulseSpacing_vs_FirstCenter_Overall.png")

# Save the figure
plt.savefig(figure_filename, dpi=300)

figure_filename = os.path.join(figures_dir, "PulseSpacing_vs_FirstCenter_Overall.pdf")
plt.savefig(figure_filename, dpi=300)

plt.show()


plt.figure(figsize=(8, 5))
plt.errorbar(npConvcenters[:-1], repetition_rates, yerr=repetition_errors, color="black", fmt='.', ecolor='red', capsize=3)
plt.xlabel(r'$\tau$ $(\mu s)$ [first center of each pair]')
plt.ylabel(r'$\Delta \tau$ (ps) [difference between centers]')
plt.title('Pulse Spacing vs τ (anchored to first center)')
plt.grid(True)
plt.tight_layout()

plt.xlim((-0.4, +0.4))

# Define the filename (standardized name inside the Figures folder)
figure_filename = os.path.join(figures_dir, "PulseSpacing_vs_FirstCenter_Zoom1.png")

# Save the figure
plt.savefig(figure_filename, dpi=300)

figure_filename = os.path.join(figures_dir, "PulseSpacing_vs_FirstCenter_Zoom1.pdf")
plt.savefig(figure_filename, dpi=300)


plt.show()


# %% Opening the CSV files from the fits !!!!

import os
import re

# folder_path0 = 'fit_results_-10psDelay_4psStepSize'
folder_path1 = 'fit_results_-10ps_TrueDelay_4psStepSize'
folder_path2 = 'fit_results_-25ps_Delay_4psStepSize'
folder_path3 = 'fit_results_-35ps_Delay_4psStepSize'
folder_path4 = 'fit_results_-50ps_Delay_5psStepSize'
folder_path5 = 'fit_results_-60ps_Delay_14psStepSize'

FolPATHS = [folder_path1, folder_path2, folder_path3, folder_path4, folder_path5]


def file_extractor(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    files_sorted = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))

    results = []
    for file in files_sorted:
        df = pd.read_csv(os.path.join(folder_path, file))
        results.append(df)
        
    return results


# Store the results in a dictionary
all_results = {}

for i, folder_path in enumerate(FolPATHS):
    results = file_extractor(folder_path)
    all_results[f"results_{i}"] = results

# %% Extract from the pandas dataframes contained in results the needed arrays

# Initialize dictionaries to store the results for each folder
centri_all = {}
FWHMs_all = {}
errFWHMs_all = {}
errCentri_all = {}

for key, results_list in all_results.items():
    centri = []
    FWHMs = []
    errFWHMs = []
    errCentri = []
    
    for df in results_list:
        centri.append(df.loc[3, "Value"])
        FWHMs.append(df.loc[2, "Value"])
        errFWHMs.append(df.loc[2, "Uncertainty"])
        errCentri.append(df.loc[3, "Uncertainty"])
    
    # Save each list in its respective dictionary
    centri_all[key] = centri
    FWHMs_all[key] = FWHMs
    errFWHMs_all[key] = errFWHMs
    errCentri_all[key] = errCentri

for key in FWHMs_all.keys():
    FWHMs = np.array(FWHMs_all[key])
    centri = np.array(centri_all[key])
    
    index_min = np.argmin(FWHMs)
    print(f"\n\n[ {key} ]")
    print(f"The index of the minimum FWHM is: {index_min}")
    print(f"The peak with the minimum FWHM ({FWHMs[index_min]} ps) is located at {centri[index_min]} ps")



# %% Plot of the FWHMs differences with the central peak, versus the number of periods of difference...

errFWHMs = np.array(errFWHMs)  # convert list to NumPy array



# Reference peak (tau = 0)
FWHM_ref = FWHMs[index_min]
err_ref = errFWHMs[index_min]

# Compute differences with respect to tau = 0 peak
diffs = FWHMs - FWHM_ref
errors = np.sqrt(errFWHMs**2 + err_ref**2)

# Optionally exclude the tau=0 peak itself from the plot (diff=0)
mask = np.arange(len(FWHMs)) != index_min

# Compute the period offsets (e.g., -5, -4, ..., 0, +1, +2, ...)
period_offsets = np.arange(len(centri)) - index_min

# Apply the same mask you already have
period_offsets_relative = period_offsets[mask]
diffs_relative = diffs[mask]
errors_relative = errors[mask]

# Plot
plt.figure(figsize=(8,5))
plt.errorbar(period_offsets_relative, diffs_relative, yerr=errors_relative, fmt='.', color='black', capsize=2, ecolor = 'grey')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Periods of Distance from τ=0')
plt.ylabel('ΔFWHM relative to τ=0 (ps)')
plt.title('Peak Width Deviation vs Period Distance')
plt.grid(True)
plt.show()




# %% General Plotting !!!
import matplotlib.pyplot as plt
import numpy as np

# Define color and label mapping for each dataset
plot_styles = {
    'results_0': {'color': 'black', 'label': '-10ps'},
    'results_1': {'color': 'blue', 'label': '-25ps'},
    'results_2': {'color': 'red', 'label': '-35ps'},
    'results_3': {'color': 'green', 'label': '-50ps'},
    'results_4': {'color': '#FF9B00', 'label': '-60ps'}
}

plt.figure(figsize=(10, 6))

# Loop over each dataset
for key in all_results.keys():
    centri = np.array(centri_all[key])
    FWHMs = np.array(FWHMs_all[key])
    errFWHMs = np.array(errFWHMs_all[key])

    # Find the index of the minimum FWHM (tau=0 peak)
    index_min = np.argmin(FWHMs)
    FWHM_ref = FWHMs[index_min]
    err_ref = errFWHMs[index_min]

    # Compute differences and uncertainties
    diffs = FWHMs - FWHM_ref
    errors = np.sqrt(errFWHMs**2 + err_ref**2)

    # Exclude the tau=0 peak itself
    mask = np.arange(len(FWHMs)) != index_min

    # Compute period offsets relative to tau=0
    period_offsets = np.arange(len(centri)) - index_min

    # Apply mask
    period_offsets_relative = period_offsets[mask]
    diffs_relative = diffs[mask]
    errors_relative = errors[mask]

    # Get color and label
    style = plot_styles[key]



    #If you want to have the plot displayed with absolute distances over the x-Axis make sure to add
    #as later shown the x array as argument of the function np.abs()
    # Plot with error bars
    plt.errorbar(
        period_offsets_relative,
        diffs_relative,
        yerr=errors_relative,
        fmt='.',
        color=style['color'],
        ecolor='grey',
        capsize=2,
        label=style['label']
    )

# Decorations
plt.axhline(0, color='gray', linestyle='--')

# Increase tick size
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


plt.xlabel('Periods of Distance from τ=0', fontsize = 15)
# plt.xlabel('Periods of Absolute Distance from τ=0', fontsize = 15)
plt.ylabel('ΔFWHM relative to τ=0 (ps)', fontsize = 15)
plt.title(r'$g_{2}(\tau)$ Peak Width Deviation vs Period Distance', fontsize = 15)
plt.legend()
# plt.xlim(-25,25)
# plt.ylim(0,1.8)
plt.grid(True)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))

# Loop over each dataset
for key in all_results.keys():
    centri = np.array(centri_all[key])
    FWHMs = np.array(FWHMs_all[key])
    errFWHMs = np.array(errFWHMs_all[key])
    errCentri = np.array(errCentri_all[key])

    ncentri = centri/ 1e6
    nerrcentri = errCentri / 1e6

    style = plot_styles[key]

    # Plot FWHM vs Center Position with error bars
    plt.errorbar(
        ncentri, FWHMs,
        xerr=nerrcentri,
        yerr=errFWHMs,
        fmt='.',
        color=style['color'],
        ecolor='grey',
        capsize=2,
        label=style['label']
    )

# Decorations
plt.xlabel(r'Peak Center Position τ ($\mu s$)', fontsize = 15)
plt.ylabel('FWHM (ps)', fontsize = 15)
plt.title(r'$g_{2}(\tau)$ FWHM vs Peak Center Position', fontsize = 15)
plt.legend()

# Increase tick size
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# ax = plt.gca()
# ax.xaxis.set_major_formatter(EngFormatter(unit='$\mu s$'))
plt.grid(True)
plt.tight_layout()
plt.show()

# %% General WL requested NOT  ZOOMED IN !!!



# Use same style mapping as before
plot_styles = {
    'results_0': {'color': 'black', 'label': '-10ps'},
    'results_1': {'color': 'blue', 'label': '-25ps'},
    'results_2': {'color': 'red', 'label': '-35ps'},
    'results_3': {'color': 'green', 'label': '-50ps'},
    'results_4': {'color': '#FF9B00', 'label': '-60ps'}
}


plt.figure(figsize=(10, 6))

for key in centri_all.keys():
    centers = np.array(centri_all[key])
    errors = np.array(errCentri_all[key])
    
    # Compute midpoints and repetition rates
    midpoints = centers[:-1]
    repetition_rates = centers[1:] - centers[:-1]
    repetition_errors = np.sqrt(errors[1:]**2 + errors[:-1]**2)
    
    style = plot_styles[key]
    
    #Conversion to better units
    nmidpoints = midpoints / 1e6
    conv_repetitionRates = (repetition_rates / 1e3)
    conv_yerr = repetition_errors / 1e3
    
    plt.errorbar(
        nmidpoints, 
        conv_repetitionRates, 
        yerr=conv_yerr, 
        fmt='.', 
        color=style['color'], 
        ecolor='grey', 
        capsize=3, 
        label=style['label']
    )

# Plot settings
plt.xlabel(r'$\tau$ ($\mu s$) [first center of each pair]', fontsize = 15)
plt.ylabel(r'$\Delta \tau$ (ns) [difference between centers]', fontsize = 15)
plt.title('Pulse Spacing vs τ', fontsize = 15)
plt.grid(True)

# Increase tick size
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend()
plt.tight_layout()
plt.show()

# %% Read CSV files for the mode locked laser

"""
For internal reference if you're analyzing the files of the last take this is the meaning of these histograms:
  Histogram 1 : Start as Sync trigger, Stop as Detection from detector 2 (TCSPC)
  Histogram 2 : Start as Sync Trigger, Stop as Detection from detector 3 (TCSPC)
  Histogram 3 : HBT detector 2 vs detector 3

"""

PATH2 = Path("ModeLockedLaser/TimeTrace_Andrea_20250604")
flder_path = PATH2
list_files = [f.name for f in flder_path.iterdir() if f.is_file() and f.suffix == '.csv']
names = [list_files[0]]  #, list_files[1], list_files[2], list_files[3], list_files[4]]

# Load the CSV file with semicolon separator


F_Index = 0


df = pd.read_csv(PATH2 / names[F_Index], sep=";") 

# Extract bin center positions (first column)
positions = df.iloc[:, 0].values

# Extract the histogram values
hist1 = df.iloc[:, 1].values
hist2 = df.iloc[:, 2].values
hist3 = df.iloc[:, 3].values
hist4 = df.iloc[:, 4].values

# Optionally extract all histograms into a 2D array
all_hists = df.iloc[:, 1:5].values
err_all_hists = np.sqrt(all_hists)



# # Optional: Print to verify
# print("Positions:", positions)
# print("Histogram 1:", hist1)
# print("All Histograms (2D):", all_hists)

# %% Start of the 2nd battle of the fits

Dataframe_Gaussianfits = []
Dataframe_Lorentzianfits = []
Dataframe_SecSquaredfits = []

INDICE = 0

Printed_Infos = True


for i in range(2):
    
    # if((i == 0) | (i == 2)):
    if(True):
        
        print(f"The figure {i} plot belongs to the hist{i+1} Histogram")
        
        # Find index of the maximum value
        peak_index = np.argmax(all_hists[:, i])
        
        # Find the position (bin center) corresponding to the peak
        peak_position = positions[peak_index]
        arr = all_hists[:, i]
        eArr = err_all_hists[:, i]
        peak_value = arr[peak_index]
        
        print(f"Peak at position: {peak_position}, with value: {peak_value}")
        
        
        #Creating a proper mask
        Fit_Distance = 100
        left = peak_position - Fit_Distance
        right = peak_position + Fit_Distance
        mask = (positions > left) & (positions < right)
        
        selected_positions = positions[mask]
        selected_Counts = arr[mask]
        selected_errors = eArr[mask]
        
        valid_mask = (
            ~np.isnan(selected_positions) &
            ~np.isnan(selected_Counts) &
            ~np.isnan(selected_errors)
        )
        selected_positions = selected_positions[valid_mask]
        selected_Counts = selected_Counts[valid_mask]
        selected_errors = selected_errors[valid_mask]
        
        
        if np.any(selected_errors <= 0):
            print("Warning: Fixing invalid error values...")
            nonzero_min = np.min(selected_errors[selected_errors > 0]) if np.any(selected_errors > 0) else 1.0
            selected_errors[selected_errors <= 0] = nonzero_min * 1e-2
        
        #Plot the histogram, including an horizontal line @ the maximum
        plt.figure(figsize=(15,8))
        plt.errorbar(positions, all_hists[:, i], yerr = err_all_hists[:, i], fmt = '.', color='grey', capsize=2, ecolor = 'orange', label = 'Data Points')
        #plt.axhline(peak_value, color = 'red')
        #plt.axvline(left, ls= '--', color = 'purple', label = 'Left Boundary')
        #plt.axvline(right, ls = '-.', color = 'purple', label = 'Right Boundary')
        
        plt.xlim(peak_position - 80, peak_position + 80)
        plt.grid(True)
        # plt.show()
        
        
        
        
        
        print('\n\nAs follows the results of the gaussian fit :\n')
        Dataframe_Gaussianfits.append(Do_Gauss_Fit_v5(selected_positions, selected_Counts, selected_errors, PrintParams=True, DebugPrints=Printed_Infos))
        
        
        
        # print('\n\nAs follows the results of the Lorentzian fit\n')
        # Dataframe_Lorentzianfits.append(Do_Lorentzian_Fit(selected_positions, selected_Counts, selected_errors, PrintParams=True, View=False, DebugPrints=Printed_Infos))
        
        
        print('\n\nAs follows the results of the sech² fit:\n')
        Dataframe_SecSquaredfits.append(Do_Sech2_Fit(selected_positions, selected_Counts, selected_errors, PrintParams=True, DebugPrints=Printed_Infos))
        
        
        
        # print(f'\n\n{INDICE}')
        plt.plot(selected_positions, gaussian(selected_positions, Dataframe_Gaussianfits[INDICE].loc[0]["Value"],
                                              Dataframe_Gaussianfits[INDICE].loc[1]["Value"],
                                              Dataframe_Gaussianfits[INDICE].loc[2]["Value"],
                                              Dataframe_Gaussianfits[INDICE].loc[3]["Value"]), color = 'red', label = "Gaussian fit", ls = '-.')
       
        # plt.plot(selected_positions, lorentzian(selected_positions, Dataframe_Lorentzianfits[INDICE].loc[0]["Value"],
        #                                       Dataframe_Lorentzianfits[INDICE].loc[1]["Value"],
        #                                       Dataframe_Lorentzianfits[INDICE].loc[2]["Value"],
        #                                       Dataframe_Lorentzianfits[INDICE].loc[3]["Value"]), color = '#003399', label = "Lorentzian fit", ls = '-.')
        
        
        plt.plot(selected_positions, sech2(selected_positions, Dataframe_SecSquaredfits[INDICE].loc[0]["Value"],
                                              Dataframe_SecSquaredfits[INDICE].loc[1]["Value"],
                                              Dataframe_SecSquaredfits[INDICE].loc[2]["Value"],
                                              Dataframe_SecSquaredfits[INDICE].loc[3]["Value"]), color = 'green', label = r"$Sech^{2}$ fit")
        
        plt.legend(fontsize=15, loc='best', frameon=True)
        
        plt.xlabel('Time [ps]')
        plt.ylabel('Counts')
        
        plt.show()
        
        # print(f'IL VALORE DELLA VARIABILE "INDICE" È ATTUALMENTE : {INDICE}')
        
        INDICE += 1
        print('\n\n\n')
        
        


# %% Saving Fits Dataframes

import os

# # Create a folder if it doesn't exist
# os.makedirs(PATH/"fit_results_-10psDelay_4psStepSize", exist_ok=True)

output_dir = f"fit_results_ModeLockedfemtoSec_file{F_Index + 1}"
os.makedirs(output_dir, exist_ok=True)  # creates the folder if it doesn't exist

# Define the Figures subfolder path
figures_dir = os.path.join(output_dir, "Images")

# Create the Figures subfolder if it doesn't exist
os.makedirs(figures_dir, exist_ok=True)

# Crea le sottocartelle per ogni tipo di fit
gaussian_dir = os.path.join(output_dir, "Gaussian")
lorentzian_dir = os.path.join(output_dir, "Lorentzian")
secsquared_dir = os.path.join(output_dir, "SecSquared")

os.makedirs(gaussian_dir, exist_ok=True)
os.makedirs(lorentzian_dir, exist_ok=True)
os.makedirs(secsquared_dir, exist_ok=True)

# Salva i DataFrame nelle rispettive sottocartelle
for i, df in enumerate(Dataframe_SecSquaredfits):
    df.to_csv(f"{secsquared_dir}/SecSquared_Results_{i}.csv", index=False)

for i, df in enumerate(Dataframe_Lorentzianfits):
    df.to_csv(f"{lorentzian_dir}/Lorentz_Results_{i}.csv", index=False)

for i, df in enumerate(Dataframe_Gaussianfits):
    df.to_csv(f"{gaussian_dir}/Gaussian_Results_{i}.csv", index=False)

# %% Opening files from the DataFrames

"""
This cell is specifically designed to open CSV files with the DataFrames from the previous cells.

Select according to the explanations later the value of TYPE to choose which fits you'd like to use and 
expect a dictionary with the resulting dataframes of each file in every page !!
this dictionary will come with the name "all_results", in which every page is then called 'results_i', 
with i being the number of the file being used.

"""


import os
import re

"""
Define the type of Dataframes that you wish to open right now :
    - Set  TYPE = 0 if you want to open Gaussian Fits Dataframes 
    - Set  TYPE = 1 if you want to open Lorenzian Fits Dataframes 
    - Set  TYPE = 2 if you want to open SecSquared Fits Dataframes 
"""

TYPE = 2

def switch_case(TYPE):
    return {
        0: "/Gaussian",
        1: "/Lorentzian",
        2: "/SecSquared"
    }.get(TYPE, "Choice of value TYPE is not valid, read the instructions better and try again !!!!!")




folder_path1 = f'fit_results_ModeLockedfemtoSec_file1{switch_case(TYPE)}'
folder_path2 = f'fit_results_ModeLockedfemtoSec_file2{switch_case(TYPE)}'
folder_path3 = f'fit_results_ModeLockedfemtoSec_file3{switch_case(TYPE)}'
folder_path4 = f'fit_results_ModeLockedfemtoSec_file4{switch_case(TYPE)}'
folder_path5 = f'fit_results_ModeLockedfemtoSec_file5{switch_case(TYPE)}'

FolPATHS = [folder_path1, folder_path2, folder_path3, folder_path4, folder_path5]


def file_extractor(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    files_sorted = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))

    results = []
    for file in files_sorted:
        df = pd.read_csv(os.path.join(folder_path, file))
        results.append(df)
        
    return results


# Store the results in a dictionary
all_results = {}

for i, folder_path in enumerate(FolPATHS):
    results = file_extractor(folder_path)
    all_results[f"results_{i}"] = results


# %% Extraction of everything from the dictionaries


#Initialization of the dictionaries

# Centers_all = {}
# Widths_all = {}
# errWidths_all = {}
# errCenters_all = {}
# Amplitudes_all = {}
# errAmplitudes_all = {}
# Baselines_all = {}
# errBaselines_all = {}


# Initialize flat/global lists
Centers_all = []
Widths_all = []
errWidths_all = []
errCenters_all = []
Amplitudes_all = []
errAmplitudes_all = []
Baselines_all = []
errBaselines_all = []


# for key, results_list in all_results.items():
#     Amplitudes = []
#     Widths = []
#     errWidths = []
#     errAmplitudes = []
#     Baselines = []
#     errBaselines = []
#     Centers = []
#     errCenters = []
    
#     for df in results_list:
#         Centers.append(df.loc[3, "Value"])
#         Widths.append(df.loc[2, "Value"])
#         errWidths.append(df.loc[2, "Uncertainty"])
#         errCenters.append(df.loc[3, "Uncertainty"])
#         Baselines.append(df.loc[1,"Value"])
#         errBaselines.append(df.loc[1, "Uncertainty"])
#         Amplitudes.append(df.loc[0, "Value"])
#         errAmplitudes.append(df.loc[0, "Uncertainty"])
    
#     # Save each list in its respective dictionary
#     Centers_all[key] = Centers
#     Widths_all[key] = Widths
#     errWidths_all[key] = errWidths
#     errCenters_all[key] = errCenters
#     Amplitudes_all[key] = Amplitudes
#     errAmplitudes_all[key] = errAmplitudes
#     Baselines_all[key] = Baselines
#     errBaselines_all[key] = errBaselines


# Loop through each folder's results
for key, results_list in all_results.items():
    for df in results_list:
        Centers_all.append(df.loc[3, "Value"])
        Widths_all.append(df.loc[2, "Value"])
        errWidths_all.append(df.loc[2, "Uncertainty"])
        errCenters_all.append(df.loc[3, "Uncertainty"])
        Baselines_all.append(df.loc[1, "Value"])
        errBaselines_all.append(df.loc[1, "Uncertainty"])
        Amplitudes_all.append(df.loc[0, "Value"])
        errAmplitudes_all.append(df.loc[0, "Uncertainty"])



for i in range(5):
    print(f'File{i+1} ', Amplitudes_all[2*i], Amplitudes_all[2*i+1], '\n\n')



# for key in Widths_all.keys():
#     Widths = np.array(Widths_all[key])
#     Centers = np.array(Centers_all[key])
#     Amplitudes = np.array(Amplitudes_all[key])
#     Baselines = np.array(Baselines_all[key])
#     errWidths = np.array(errWidths_all[key])
#     errCenters = np.array(errCenters_all[key])
#     errAmplitudes = np.array(errAmplitudes_all[key])
#     errBaselines = np.array(errBaselines_all[key])


# %% The direct convolution integral using hypothetical datas

from scipy.integrate import quad
import matplotlib.pyplot as plt





#What i see in the Lab measurements
x = np.linspace(-700, 700, 25000)
f1 = sech2(x, Amplitudes_all[0], Baselines_all[0], Widths_all[0], 0)



def model(t, pulse_fwhm, waveguide_fwhm, irf_fwhm):
    pulse = sech2(t, pulse_fwhm)
    dispersion = gaussian(t, waveguide_fwhm)
    irf = gaussian(t, irf_fwhm)

    # Convolve step by step
    intermediate = np.convolve(pulse, dispersion, mode='same')
    full = np.convolve(intermediate, irf, mode='same')

    return full









# plt.figure()
# plt.plot(x, f1)
# plt.show()




# %% Attempt to deconvolve via numerical simulations 

"""
For internal reference if you're analyzing the files of the last take this is the meaning of these histograms:
  Histogram 1 : Start as Sync trigger, Stop as Detection from detector 2 (TCSPC)
  Histogram 2 : Start as Sync Trigger, Stop as Detection from detector 3 (TCSPC)
  Histogram 3 : HBT detector 2 vs detector 3

"""
from scipy.optimize import curve_fit
PATH2 = Path("ModeLockedLaser/TimeTrace_Andrea_20250604")
flder_path = PATH2
list_files = [f.name for f in flder_path.iterdir() if f.is_file() and f.suffix == '.csv']
names = [list_files[0]]  #, list_files[1], list_files[2], list_files[3], list_files[4]]

# Load the CSV file with semicolon separator


F_Index = 0


df = pd.read_csv(PATH2 / names[F_Index], sep=";") 

# Extract bin center positions (first column)
positions = df.iloc[:, 0].values

# Extract the histogram values
hist1 = df.iloc[:, 1].values
hist2 = df.iloc[:, 2].values
hist3 = df.iloc[:, 3].values
hist4 = df.iloc[:, 4].values

# Optionally extract all histograms into a 2D array
all_hists = df.iloc[:, 1:5].values
err_all_hists = np.sqrt(all_hists)




Dataframe_ModelFits = []
Dataframe_SecSquaredfits = []

INDICE = 0

Printed_Infos = True


def sech2_fwhm(x, A, b, fwhm, t0):
    tau = fwhm / 1.76
    return A * (1 / np.cosh((x - t0) / tau))**2 + b



for i in range(2):
    # if i in [0, 2]:
        
        print(f"The figure {i} plot belongs to the hist{i+1} Histogram")
        
        # Extract data
        arr = all_hists[:, i]
        eArr = err_all_hists[:, i]
        peak_index = np.argmax(arr)
        peak_position = positions[peak_index]
        
        #Centering everything in t=0
        positions -= peak_position
        peak_position -= peak_position
        
        peak_value = arr[peak_index]
        
        # Define fit window
        Fit_Distance = 100
        left, right = peak_position - Fit_Distance, peak_position + Fit_Distance
        mask = (positions > left) & (positions < right)
        
        selected_positions = positions[mask]
        selected_Counts = arr[mask]
        selected_errors = eArr[mask]
        
        
        # --- Fit with sech² only
        print('\n\nAs follows the results of the sech² fit of the TCSPC Histogram:\n')
        df_sech2 = Do_Sech2_Fit(selected_positions, selected_Counts, selected_errors, PrintParams=True, DebugPrints=Printed_Infos)
        Dataframe_SecSquaredfits.append(df_sech2)

        # Extract initial guesses from sech² fit
        A0 = df_sech2.loc[0]["Value"]
        b0 = df_sech2.loc[1]["Value"]
        t0 = df_sech2.loc[3]["Value"]
        fwhm_pulse0 = df_sech2.loc[2]["Value"] * 1.76  # tau → FWHM

        t_min, t_max = selected_positions.min(), selected_positions.max()
        t_range = t_max - t_min

        # Plot sech²-only fit
        # plt.plot(selected_positions, sech2(selected_positions, A0, b0, df_sech2.loc[2]["Value"], t0),
                 # color='green', label=r"Sech$^2$ Fit")

        # --- Convoluted Model  -----> to be upgraded to the implementation via np.quad()
        def convolved_model(x, A, b, fwhm_pulse, t0, fwhm_irf):
            pulse = sech2_fwhm(x, A, 0, fwhm_pulse, t0)  # no baseline
            irf = gaussian(x, 1, 0, fwhm_irf, t0)  # unit amplitude, centered at t0
            area = np.trapezoid(irf, x)
            if area > 0:
                irf /= area  # normalize IRF
            conv = np.convolve(pulse, irf, mode='same')
            return conv + b

        # Guesses and bounds
        # guess = [A0, b0, fwhm_pulse0, t0, 10.0]  # 10 ps IRF guess
        guess = [A0, b0, 7, t0, 50]  # 10 ps IRF guess
        # lower_bounds = [0, 0, 1e-3, t_min - 0.2 * t_range, 1e-3]
        # upper_bounds = [np.inf, np.inf, t_range, t_max + 0.2 * t_range, 0.5 * t_range]
        
        # Lower and upper bounds for the fit
        lower_bounds = [
            0,              # A (amplitude)
            0,              # b (baseline)
            0.002,           # fwhm_pulse ≥ 20 fs
            t_min - 0.1 * t_range,  # t0: small shift allowed
            0             # fwhm_irf ≥ 5 ps, conservative lower bound
        ]
        
        upper_bounds = [
            np.inf,         # A
            np.inf,         # b
            100,          # fwhm_pulse ≤ 200 fs
            t_max + 0.1 * t_range,  # t0
            200.0           # fwhm_irf ≤ 200 ps
        ]


        # params = [A0, b0, 0.05, t0, 27]
        # for p, low, high, name in zip(params, lower_bounds, upper_bounds, ["A0", "b0", "tau0", "t0", "irf0"]):
        #     if not (low <= p <= high):
        #         print(f"❌ Parameter '{name}' = {p} is outside bounds: [{low}, {high}]")


        try:
            popt, pcov = curve_fit(
                convolved_model,
                selected_positions,
                selected_Counts,
                p0=guess,
                bounds=(lower_bounds, upper_bounds),
                sigma=selected_errors,
                absolute_sigma=True
            )
        except RuntimeError as e:
            print("Fit failed:", e)
            continue

        # Compute fit residuals
        model_vals = convolved_model(selected_positions, *popt)
        residuals = selected_Counts - model_vals

        # Chi-squared
        chi2 = np.sum((residuals / selected_errors)**2)

        # Degrees of freedom
        dof = len(selected_positions) - len(popt)

        # Reduced chi-squared
        red_chi2 = chi2 / dof if dof > 0 else np.nan
        

        # Save parameters
        fit_result = pd.DataFrame({
            "Parameter": ["A", "b", "fwhm_pulse", "t0", "fwhm_irf"],
            "Value": popt,
            "StdDev": np.sqrt(np.diag(pcov))
        })
        
        # Append chi-squared stats to DataFrame
        fit_result = pd.concat([
            fit_result,
            pd.DataFrame({
                "Parameter": ["Chi2", "Reduced_Chi2"],
                "Value": [chi2, red_chi2],
                "StdDev": [np.nan, np.nan]
            })
        ], ignore_index=True)
        
        Dataframe_ModelFits.append(fit_result)
        
        print('\n\nAs follows the results of the datapoints fit with the convoluted model : \n\n')
        print(fit_result)
        
        # print(f'\n\nReduced Chi squared currently of {red_chi2}')
        
        # Plot original histogram
        plt.figure(figsize=(15, 8))
        plt.errorbar(positions , arr, yerr=eArr, fmt='.', color='grey', capsize=2, ecolor='orange', label='Data')
        plt.grid(True)

        # Plot IRF and convolved fit
        irf_curve = gaussian(selected_positions, 1, 0, popt[4], popt[3])
        irf_curve /= np.trapezoid(irf_curve, selected_positions)
        irf_curve_scaled = irf_curve * popt[0]  # scale to pulse amplitude

        plt.plot(selected_positions, irf_curve_scaled + popt[1], '--', color='purple', label="IRF (Gaussian)")
        # plt.plot(selected_positions, irf_curve + popt[1], '--', color='purple', label="IRF (Gaussian)")

        plt.plot(selected_positions, convolved_model(selected_positions, *popt),
                 '-', color='red', label="Convolved Fit")
        
        optical_pulse = sech2_fwhm(selected_positions , popt[0], 0, popt[2], popt[3])
        plt.plot(selected_positions , optical_pulse + popt[1], '--', color='blue', label='Optical Pulse (Sech²)')
        
        plt.xlim(peak_position - 80, peak_position + 80)
        plt.ylim(-25, peak_value + 3600)
        
        plt.xlabel('Time [ps]')
        plt.ylabel('Counts')
        plt.legend(fontsize=14)
        plt.title(f"Fit for Histogram {i+1}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        INDICE += 1
        print("\n\n")


# %% Pulse Broadening Calculator

def GVD_PulseBroadening_Sech2Shape(INPUT_FWHM, CableLength):
    """
    Prints Out pulse Broadening that changes first argument 'INPUT_FWHM' inserted in [fs], into a broader value due to GVD inside the fiber
    """
    # beta_2 = 38.3 # [ps^2 / km]
    # beta_2 = 43 # [ps^2 / km]
    beta_2 = 34.7 # [ps^2 / km]
    beta_2_fskm = beta_2 * 1e6 #[fs^2 / km]
    # tau_0 = 56.82*1e-15 # [fs]
    tau_0 = (INPUT_FWHM / 1.76) #[fs]
    
    
    Ld_km = (tau_0**2) / (beta_2_fskm)  
    Ld_cm = Ld_km * 1e5
    
    return (tau_0 * 1e-15 * np.sqrt(1+(CableLength/Ld_cm)**2)) * 1.76 * 1e+12

print('The FWHM of the optical pulses after ~15m of HP780 fiber is :', GVD_PulseBroadening_Sech2Shape(50.3, 1500), '[ps]')
# %% Part2
import numpy as np

def GVD_780HP_Sech2Broadening(input_fwhm_fs, fiber_length_cm, wavelength_nm):
    """
    Estimates output pulse FWHM [ps] after propagation through Thorlabs 780HP fiber,
    accounting for GVD at specified wavelength.
    
    Parameters:
    - input_fwhm_fs : float
        Input pulse FWHM duration in femtoseconds [fs]
    - fiber_length_cm : float
        Fiber length in centimeters [cm]
    - wavelength_nm : float
        Wavelength of operation in nanometers [nm]
        
    Returns:
    - output_fwhm_ps : float
        Output pulse FWHM in picoseconds [ps]
    """
    # Approximate wavelength-dependent beta_2 [ps^2/km]
    if wavelength_nm <= 850:
        beta_2 = 38.3
    elif wavelength_nm >= 950:
        beta_2 = 23.0
    else:
        # Linear interpolation between 850 and 950 nm
        beta_2 = 38.3 + (23.0 - 38.3) * (wavelength_nm - 850) / (950 - 850)
    
    beta_2_fs2_per_km = beta_2 * 1e6  # convert to fs^2/km

    tau_0_fs = input_fwhm_fs / 1.76
    Ld_km = tau_0_fs**2 / beta_2_fs2_per_km
    Ld_cm = Ld_km * 1e5

    output_tau_fs = tau_0_fs * np.sqrt(1 + (fiber_length_cm / Ld_cm)**2)
    output_fwhm_fs = output_tau_fs * 1.76
    output_fwhm_ps = output_fwhm_fs / 1000  # convert fs to ps

    return output_fwhm_ps

result = GVD_780HP_Sech2Broadening(45, 1500, 897)
print(f"Output FWHM: {result:.3f} ps")

# %% Following WL new orders : HUGE Retreat to understand better what we're doing !!!


"""
PLEASE REFER TO PLOTS_TSUNAMI FOLDER AND RUN THE PROGRAM THAT YOU FIND INSIDE THERE TO RUN THIS PART OF THE SCRIPT !!
HERE YOU WON'T FIND THE LATEST VERSION AVAILABLE !!!'

For internal reference if you're analyzing the files of the last take this is the meaning of these histograms:
  Histogram 1 : Start as Sync trigger, Stop as Detection from detector 2 (TCSPC)
  Histogram 2 : Start as Sync Trigger, Stop as Detection from detector 3 (TCSPC)
  Histogram 3 : HBT detector 2 vs detector 3

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pathlib import Path

# --- File setup ---
PATH2 = Path("ModeLockedLaser/TimeTrace_Andrea_20250604")
flder_path = PATH2
list_files = [f.name for f in flder_path.iterdir() if f.is_file() and f.suffix == '.csv']
names = [list_files[0]]

# --- Load data ---
F_Index = 0
df = pd.read_csv(PATH2 / names[F_Index], sep=";") 

# --- Extract bin centers and histograms ---
positions = df.iloc[:, 0].values
hist1 = df.iloc[:, 1].values
hist2 = df.iloc[:, 2].values
hist3 = df.iloc[:, 3].values

# --- Errors from shot noise ---
ehist1 = np.sqrt(hist1)
ehist2 = np.sqrt(hist2)
ehist3 = np.sqrt(hist3)

# --- Peak detection ---
peak_index1 = np.argmax(hist1)
peak_index2 = np.argmax(hist2)
peak_index3 = np.argmax(hist3)

peak_position1 = positions[peak_index1]
peak_position2 = positions[peak_index2]
peak_position3 = positions[peak_index3]

# --- Center time axes ---
positions1 = positions.copy() - peak_position1
positions2 = positions.copy() - peak_position2
positions3 = positions.copy() - peak_position3

# --- Function to compute FWHM and return FWHM info ---
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

# --- Compute FWHM for each histogram ---
x_fine1, y_fine1, xL1, xR1, hmax1, fwhm1 = compute_fwhm(positions1, hist1)
x_fine2, y_fine2, xL2, xR2, hmax2, fwhm2 = compute_fwhm(positions2, hist2)
x_fine3, y_fine3, xL3, xR3, hmax3, fwhm3 = compute_fwhm(positions3, hist3)

# --- Plot ---
plt.figure(figsize=(18, 8))

# Histogram 1
plt.errorbar(positions1, hist1, yerr=ehist1, fmt='*', color='black',
             capsize=2, ecolor='orange', label='TCSPC SYNC vs DET2')
plt.plot(x_fine1, y_fine1, 'k--', alpha=0.5)
plt.axhline(hmax1, color='black', linestyle='--', alpha=0.25)
plt.axvline(xL1, color='black', linestyle='--', alpha=0.25)
plt.axvline(xR1, color='black', linestyle='--', alpha=0.25)
plt.hlines(hmax1, xL1, xR1, colors='black', linewidth=2,
           label=f'FWHM Sync vs Det2 = {fwhm1:.2f}', alpha=0.7)

# Histogram 2
plt.errorbar(positions2, hist2, yerr=ehist2, fmt='.', color='brown',
             capsize=2, ecolor='orange', label='TCSPC SYNC vs DET3')
plt.plot(x_fine2, y_fine2, 'brown', linestyle='--', alpha=0.5)
plt.axhline(hmax2, color='brown', linestyle='--', alpha=0.25)
plt.axvline(xL2, color='brown', linestyle='--', alpha=0.25)
plt.axvline(xR2, color='brown', linestyle='--', alpha=0.25)
plt.hlines(hmax2, xL2, xR2, colors='brown', linewidth=2,
           label=f'FWHM Sync vs Det3 = {fwhm2:.2f}', alpha=0.7)

# Histogram 3
plt.errorbar(positions3, hist3, yerr=ehist3, fmt='.', color='green',
             capsize=2, ecolor='orange', label='HBT Detector2 vs Detector3')
plt.plot(x_fine3, y_fine3, 'green', linestyle='--', alpha=0.5)
plt.axhline(hmax3, color='green', linestyle='--', alpha=0.25)
plt.axvline(xL3, color='green', linestyle='--', alpha=0.25)
plt.axvline(xR3, color='green', linestyle='--', alpha=0.25)
plt.hlines(hmax3, xL3, xR3, colors='green', linewidth=2,
           label=f'FWHM HBT = {fwhm3:.2f}', alpha=0.7)

# --- Plot setup ---
plt.xlim(-400, +700)
plt.ylabel("Counts (a.u.)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


