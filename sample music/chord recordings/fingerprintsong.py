#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:25:10 2018

@author: Jay
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from scipy.io import wavfile

from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, iterate_structure
from scipy.ndimage.filters import maximum_filter

### FILENAME without .wav ###
filename = 'Gmin3'

# read .wav file
sample_rate, samples = wavfile.read(filename+'.wav')

# if stereo track, average the 2 channels
size = np.shape(samples)
if len(size) > 1:
    if size[1] > 1:
        samples = np.mean(samples,axis=1)
        #wavfile.write('monoavg.wav', sample_rate, samples)

#arr2D = mlab.specgram(samples,
#                      NFFT=4096,
#                      Fs=44100,
#                      window=mlab.window_hanning,
#                      noverlap=int(4096 * 0.5))[0]
arr2D,arr2D_freqs,arr2D_t = mlab.specgram(samples,
                      NFFT=4096,
                      Fs=44100,
                      window=mlab.window_hanning,
                      noverlap=int(4096 * 0.5))
deltaT = arr2D_t[1] - arr2D_t[0]
deltaF = arr2D_freqs[1] - arr2D_freqs[0]

arr2D = 10 * np.log10(arr2D)
arr2D[arr2D == -np.inf] = 0

struct = generate_binary_structure(2, 1)
neighborhood = iterate_structure(struct, 20)
local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
background = (arr2D == 0)
eroded_background = binary_erosion(background, structure=neighborhood,border_value=1)
detected_peaks = local_max ^ eroded_background
amps = arr2D[detected_peaks]
j, i = np.where(detected_peaks)
amps = amps.flatten()
peaks = zip(i, j, amps)
peaks_filtered = [x for x in peaks if x[2] > 10] # freq, time, amp
frequency_idx = [x[1] for x in peaks_filtered]
time_idx = [x[0] for x in peaks_filtered]

# plots
fig, ax = plt.subplots()
ax.imshow(arr2D)
ax.scatter(time_idx, frequency_idx, s=2**2)
ax.set_xlabel('Time')
ax.set_ylabel('Frequency')
ax.set_title("Spectrogram")
plt.gca().invert_yaxis()
plt.show()

plt.figure(2)
plt.clf()
plt.scatter(time_idx, frequency_idx, s=2**2)

## save data
np.savetxt(filename+'_values.txt', np.c_[time_idx, frequency_idx], fmt='%10.1f')
np.savetxt(filename+'_deltas.txt', np.c_[deltaT, deltaF], fmt='%10.5f')