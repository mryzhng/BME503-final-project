#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:25:10 2018

@author: Jay
"""



##===============================================================================================##
## ------------------------------------- THINGS TO NOTE ---------------------------------------- ##

## FOR OPTIMIZATION OF CODE
# chance Io which is the maximum input current value that controls how many times the neurons are firing in 100ms
# each note or time point is of 100 ms, so neurons should fire enough times in those 100 to be able to add noise
# adding poisson noise will go in input current to the neurons, amplitude range of noise will depend on what you pick for Io.

## --------------------------------------------------------------------------------------------- ##
##===============================================================================================##








##=========================================================================##
## -------------- SPECTROGRAM & AUDIO FINGERPRINT (PEAKS) ---------------- ##
##=========================================================================##

## All libraries
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, iterate_structure
from scipy.ndimage.filters import maximum_filter
from brian2 import *
import matplotlib.pyplot as plt
import random
import numpy as np   
# %matplotlib qt


### FILENAME without .wav ###
# edit here if new song 
filename = 'samplemusic/cello'

# read .wav file
sample_rate, samples = wavfile.read(filename+'.wav')

# if stereo track, average the 2 channels
size = np.shape(samples)
if len(size) > 1:
    if size[1] > 1:
        samples = np.mean(samples,axis=1)
        #wavfile.write('monoavg.wav', sample_rate, samples)
# spectrogram
arr2D,arr2D_freqs,arr2D_t = mlab.specgram(samples,
                      NFFT=4096,
                      Fs=44100,
                      window=mlab.window_hanning,
                      noverlap=int(4096 * 0.5))
deltaT = arr2D_t[1] - arr2D_t[0]
deltaF = arr2D_freqs[1] - arr2D_freqs[0]
# log transform & remove infs
arr2D = 10 * np.log10(arr2D)
arr2D[arr2D == -np.inf] = 0
# local max filter to find peaks
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

# multiply time & frequency by appropriate value
times_orig = [x*deltaT for x in time_idx]
peakfreqs = [x*deltaF for x in frequency_idx] # number of peaks in the song

# round times to nearest 0.1
peaktimes = [round(elem, 1) for elem in times_orig] 

# create array of all 0.1s intervals in song
song_times = np.arange(0,max(peaktimes)+0.1,0.1) # time tracking 






##=========================================================================##
##------------------------------ TUNING CURVE -----------------------------##
##=========================================================================##


## Simulation ------------------------------
songfreq = np.random.randint(20,20000, size=(6,5)) # input frequencies to the song
#inputfreq = array([460, 2055, 9200, 15512, 16730, 19788])

## Initialize variables for front end sensory system

N = 1000 # number of neurons to simulate
order = math.ceil(sqrt(N)) # for plotting purposes
#tuning_frequencies = [220.0 , 246.9 , 261.6 , 293.7 , 329.6 , 349.2 , 392.0] 
tuning_frequencies = linspace(20,20000,N) # prefered frequencies for neuron group 
 
Io=5 # maximum firing rate at prefered frequency
at= 0.99# width parameter 


# Izhikevich LTS neurons parameters
a = 0.02  
b = 0.2
c = -65
d = 0.5

tau_syn=1.0 # (from bug model)
g_synpk=0.4
g_synmaxval=(g_synpk/(tau_syn*exp(-1)))
Esyn = 40


## Defining functions ##

# tuning curve function
def neuron_tuning(s,fpref):
    lengths=size(s) # s is the x vector or different frequency values in a range going from -p 
    current=0.0*linspace(-pi,pi,lengths) # initialize dummy array to replace with firing value = used as current input
    for i in range(0,lengths):
       if (cos(s[i]-fpref) > at and abs(s[i]-fpref)<pi): # sets the condition for non 0 inputs 
           # has to be set at > at to make sure we get a only upper half of a sine wave
           current[i]=Io/(1-at)*(cos(s[i]-fpref)-at) # cos(s-spref) should yield cos(0)=1 max value when s = spref
       else:
           current[i]=0
    return current

def f2rad(freq, fmin, fmax): # give out radian anlge for frequency (from 0 to 2pi)
    frange = fmax - fmin
    fdiff = freq - fmin
    fradian = (fdiff/frange)*2*pi
    return fradian

## Izhikevich equations ##
neuron_eqs = '''
fchar : 1
I : 1

# Izhikevich model
dv/dt = (0.04*(v**2) + 5*v + 140 - u + I + g*(Esyn-v))/ms : 1
du/dt = a*(b*v - u)/ms : 1

# Alpha synapse model
dz/dt = (-z / tau_syn)/ms : 1
dg/dt = ((-g/tau_syn) + z)/ms : 1
'''
neuron_reset = '''
v = c
u = u + d
'''


## Defining sensory neuron group -----------
sensory_neurons = NeuronGroup(N, neuron_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = neuron_reset,method='euler')
sensory_neurons.v = c # same Izhi equations
sensory_neurons.u = c*b
sensory_neurons.fchar = tuning_frequencies

## Running Monitors ----------------------
# SM SPM for the group of neurons  
SN = StateMonitor(sensory_neurons, ('v','I'), record=True) # state monitor for action potential tracking
SPM = SpikeMonitor(sensory_neurons) # recording spikes

numfreq = 0
## LOOP FOR ITERATING THROUGH song_times
for t in range(0,len(song_times)):
    #plt.figure()
    #inputfreq = songfreq.T[f]
    
    # get frequencies at the current timepoint
    inputfreq = [peakfreqs[x] for x in range(0,len(peaktimes)) if peaktimes[x]==round(song_times[t],1)]
    #numfreq = numfreq + len(inputfreq)
    #print(song_times[t], numfreq, inputfreq)

    # current input
    # reset currents from prev iter back to 0
    for n in range(0,N):
        sensory_neurons.I[n] = 0
    # add current based on inputfreq
    for i in range (0, len(inputfreq)): # outputting 
        for n in range (0, N):
            sensory_neurons.I[n] = sensory_neurons.I[n] + neuron_tuning(f2rad(linspace(inputfreq[i],inputfreq[i],1),220,440),f2rad(sensory_neurons.fchar[n],220,440))
## ------------------- ADD POISSON NOISE HERE TO INPUT CURRENT TO SENSORY NEURONS ------------------ ##
    
    ## Run
    run(100*ms,report='text') 
    # 100 ms is equal to the duration of play of each tone in the song (is equal to temporal resolution in fingerprinting code)
    

## PLOTTING
figure(1)
clf()
plot(SPM.t/ms, SPM.i, 'k.')
ylim((-10, N+10))

## For plotting tuning curves - NOT useful if N>50
#s = np.linspace(20,20000,10000)
#figure(1)
#for n in range (0, N):
#   curve = neuron_tuning(f2rad(s,20,20000),f2rad(sensory_neurons.fchar[n],20,20000))
#   hold(True)
#   plot(s,curve)
#   ylim((-1, 7))   