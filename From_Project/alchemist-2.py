"""
Created on Mon Apr 28 14:26:36 2025

@author: Maccarinelli
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar
import numpy.typing as npt
from pathlib import Path



# %% 1st part of the script !!!!!!!!!!!
PATH = Path("RAW")
folder_path = PATH
file_list = [f.name for f in folder_path.iterdir() if f.is_file() and f.suffix == '.bin']
names = [file_list[0], file_list[1], file_list[2], file_list[3], file_list[4], file_list[5], file_list[6], file_list[7]]

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

coincidence_counts, taus = get_coincidence_counts_from_files(fname_detector2, fname_detector3, stepsize_ps=STEP_ps, maxtime_ps=MaxTauDistance)

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

# plt.figure()
# plt.title(f"Second Order Correlation Function for -10ps delay pulses and binsize {STEP_ps}ps")
# plt.plot(taus, Ncounts, linewidth=0.8, color='#FF9B00', label = 'Line')  
# plt.scatter(taus, Ncounts, s=4, color='black', label = 'Data Points')

# # Axis labels bigger
# plt.xlabel(r"$\Delta \tau$ [ps]", fontsize=16)
# plt.ylabel(r"$g_{2}(\tau)$", fontsize=16)

# # Increase tick size
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=13)

# # Legend (if you want it)
# plt.legend(fontsize=14, loc='best', frameon=True)
# plt.xlim((10250, +55000))
# plt.ylim((-3, +150))
# plt.tight_layout()
# plt.show()



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
plt.xlim((-4900, -4600))
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
    Right_Borders = np.arange(lim_right_zero, 1.5e+06, Pulser_periodicity)
    Left_Borders = np.arange(lim_left_zero, 1.5e+06, Pulser_periodicity)
    
    ### Negative direction
    Left_Borders_Opp = np.arange(lim_left_zero - Pulser_periodicity, -1.5e+06, -Pulser_periodicity)
    Right_Borders_Opp = np.arange(lim_right_zero - Pulser_periodicity, -1.5e+06, -Pulser_periodicity)
    
    # Reverse the negative direction arrays
    Left_Borders_Opp_reversed = Left_Borders_Opp[::-1]
    Right_Borders_Opp_reversed = Right_Borders_Opp[::-1]
    
    # Concatenate the positive and negative direction intervals
    ALL_LEFT_BORDERS = np.concatenate([Left_Borders_Opp_reversed, Left_Borders])
    ALL_RIGHT_BORDERS = np.concatenate([Right_Borders_Opp_reversed, Right_Borders])
    
    return ALL_LEFT_BORDERS, ALL_RIGHT_BORDERS
    
    
    
I_primo = (-4895, -4630)

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
plt.xlim((-155000, 0))
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

    
    
    

plt.errorbar(centri, FWHMs, xerr = errCentri, yerr=errFWHMs, fmt='o', color='black', label='Data Points', markersize=2, elinewidth=1, capsize=1.5, ecolor='red')
plt.xlabel('Taus [ps]')
plt.ylabel('FWHMs [ps]')
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


from src.Librerie import Do_Gauss_Fit_v4

#Correction of the taus array
taus = taus+Offset

#Correction of the boundary arrays
LEFT = LEFT + Offset
RIGHT = RIGHT + Offset

# %% 5th part of the script : Visual check that the intervals' boundaries have correctly shifted !!
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

output_dir = f"fit_results_-50psDelay_{STEP_ps}psStepSize"
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

    
    
    
plt.figure(figsize=(10,6))
plt.errorbar(centri, FWHMs, xerr = errCentri, yerr=errFWHMs, fmt='o', color='black', label='Data Points', markersize=2, elinewidth=1, capsize=1.5, ecolor='red')
plt.xlabel('Taus [ps]')
plt.ylabel('FWHMs [ps]')
plt.tight_layout()

# Define the filename (standardized name inside the Figures folder)
figure_filename = os.path.join(figures_dir, "FWHM_vs_Taus.png")

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


npcenters = np.array(centers)
nperrors = np.array(errCenters)

# Midpoints (x-axis)
midpoints = (npcenters[1:] + npcenters[:-1]) / 2

# Differences (y-axis)
repetition_rates = npcenters[1:] - npcenters[:-1]
repetition_errors = np.sqrt(nperrors[1:]**2 + nperrors[:-1]**2)

plt.figure(figsize=(8, 5))
plt.errorbar(npcenters[:-1], repetition_rates, yerr=repetition_errors, color="black", fmt='.', ecolor='red', capsize=3)
plt.xlabel(r'$\tau$ (ps) [first center of each pair]')
plt.ylabel(r'$\Delta \tau$ (ps) [difference between centers]')
plt.title('Pulse Spacing vs τ (anchored to first center)')
plt.grid(True)
plt.tight_layout()

# Define the filename (standardized name inside the Figures folder)
figure_filename = os.path.join(figures_dir, "PulseSpacing_vs_FirstCenter_Overall.png")

# Save the figure
plt.savefig(figure_filename, dpi=300)
plt.show()


plt.figure(figsize=(8, 5))
plt.errorbar(npcenters[:-1], repetition_rates, yerr=repetition_errors, color="black", fmt='.', ecolor='red', capsize=3)
plt.xlim((-250000, +250000))
plt.xlabel(r'$\tau$ (ps) [first center of each pair]')
plt.ylabel(r'$\Delta \tau$ (ps) [difference between centers]')
plt.title('Pulse Spacing vs τ (anchored to first center)')
plt.grid(True)
plt.tight_layout()

# Define the filename (standardized name inside the Figures folder)
figure_filename = os.path.join(figures_dir, "PulseSpacing_vs_FirstCenter_Zoom1.png")

# Save the figure
plt.savefig(figure_filename, dpi=300)
plt.show()


# %% Opening the CSV files from the fits !!!!
import pandas as pd
import os
import re

folder_path0 = 'fit_results_-10psDelay_4psStepSize'
folder_path1 = 'fit_results_-10ps_V2_Delay_4psStepSize'
folder_path2 = 'fit_results_-25psDelay_4psStepSize'
folder_path3 = 'fit_results_-50psDelay_5psStepSize'

FolPATHS = [folder_path0, folder_path1, folder_path2, folder_path3]


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
    'results_1': {'color': 'blue', 'label': '-10ps V2'},
    'results_2': {'color': 'red', 'label': '-25ps'},
    'results_3': {'color': 'green', 'label': '-50ps'}
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
plt.xlabel('Periods of Distance from τ=0')
plt.ylabel('ΔFWHM relative to τ=0 (ps)')
plt.title('Peak Width Deviation vs Period Distance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


