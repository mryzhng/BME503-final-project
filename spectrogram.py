import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure,
                                      iterate_structure, binary_erosion)
import hashlib
from operator import itemgetter

from scipy import signal
from scipy.io import wavfile

samplingFrequency, signalData = wavfile.read('blurred_lines_short.wav')
signalData = signalData[:, 0]

arr2D = mlab.specgram(
        signalData,
        NFFT=4096,
        Fs=44100,
        window=mlab.window_hanning,
        noverlap=int(4096 * 0.5))[0]

# apply log transform since specgram() returns linear array
arr2D = 10 * np.log10(arr2D)
arr2D[arr2D == -np.inf] = 0  # replace infs with zeros

def get_2D_peaks(arr2D, plot=False, amp_min=10):
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphology.iterate_structure.html#scipy.ndimage.morphology.iterate_structure
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, 20)

    # find local maxima using our fliter shape
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)

    # Boolean mask of arr2D with True at peaks
    detected_peaks = local_max ^ eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    j, i = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()
    peaks = zip(i, j, amps)
    peaks_filtered = [x for x in peaks if x[2] > amp_min]  # freq, time, amp

    # get indices for frequency and time
    frequency_idx = [x[1] for x in peaks_filtered]
    time_idx = [x[0] for x in peaks_filtered]

    if plot:
        # scatter of the peaks
        fig, ax = plt.subplots()
        ax.imshow(arr2D)
        ax.scatter(time_idx, frequency_idx)
        ax.set_xlabel('Time')
        #ax.xlim(0, max(frequency_idx))
        ax.set_ylabel('Frequency')
        ax.set_title("Spectrogram")
        plt.gca().invert_yaxis()
        plt.show()

    return zip(frequency_idx, time_idx)

local_maxima = get_2D_peaks(arr2D, plot=True, amp_min=10)

frequencies, times = zip(*local_maxima)
plt.figure(2)
plt.clf()
plt.scatter(times,frequencies)

plt.figure(3)
plt.clf()
plt.scatter(frequencies,times)

plt.figure(4)
plt.clf()
plt.plot(local_maxima,'ko')
plt.ylim(0,200)


#plt.figure(1)
#plt.clf()
#plt.hold(True)
#plt.plot(local_maxima,'ko')