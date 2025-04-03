#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:40:07 2025

@author: andreamaccarinelli
"""

import matplotlib.pyplot as plt
import numpy as np
import numba
from numba import njit, prange
from numba_progress import ProgressBar
import numpy.typing as npt

@njit
def sparse_difference_matrix(stream1, stream2, max_difference, j_start=0):
    time_differences = []  # Usa una lista invece di un array fisso
    temp_differences = np.empty(stream2.size)

    for i in range(len(stream1)):
        for store_index, j in enumerate(range(j_start, len(stream2))):
            temp_differences[store_index] = stream2[j] - stream1[i]
            if abs(temp_differences[store_index]) > max_difference:
                if temp_differences[store_index] < 0:
                    j_start = j + 1
                else:
                    break
        time_differences.extend(temp_differences[:store_index])  # Aggiungi solo valori validi
    return np.array(time_differences, dtype=np.int64), j_start




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

coincidence_counts, taus, chunktimes = get_coincidence_counts_from_files("/Users/andreamaccarinelli/Desktop/LION/RAW_DATA/EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C2_2025-03-26T11_50_30.bin", "/Users/andreamaccarinelli/Desktop/LION/RAW_DATA/EOM_0ps_pulse_length_HBT_and_StartStop-15.200ns_reptime_C3_2025-03-26T11_50_30.bin", stepsize_ps=1000, maxtime_ps=1000000000)