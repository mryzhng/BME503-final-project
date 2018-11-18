#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 23:28:33 2018

@author: Jay
"""

from brian2 import *
import matplotlib.pyplot as plt

# input freq
inputfreq = 240

# tuning curve function
def neuron_tuning(si,spref):
    K=1
    g=30
    sigma=.8
    at= .7;
    lengthsi=size(si)
    tuneout=0.0*linspace(-pi,pi,lengthsi)
    for i in range(0,lengthsi):
       if cos(si[i]-spref) >at:
           tuneout[i]=(g/(1-at))*(cos(si[i]-spref)-at)
       else:
           tuneout[i]=0
    return tuneout

def freq2angle(freq, fmin, fmax):
    frange = fmax - fmin
    fdiff = freq - fmin
    fradian = (fdiff/frange)*2*pi
    return fradian

# Izhikevich LTS neurons
a = 0.02
b = 0.2
c = -65
d = 0.5

# from bug model
tau_syn=1.0
g_synpk=0.4
g_synmaxval=(g_synpk/(tau_syn*exp(-1)))
Esyn = 40

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

nA3 = NeuronGroup(1, neuron_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = neuron_reset,method='euler')
nA3.v = c
nA3.u = c*b
nA3.fchar = 220.0
nA3.I = neuron_tuning(freq2angle(linspace(inputfreq,inputfreq,1),220,440),freq2angle(nA3.fchar,220,440))

nB3 = NeuronGroup(1, neuron_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = neuron_reset,method='euler')
nB3.v = c
nB3.u = c*b
nB3.fchar = 246.9
nB3.I = neuron_tuning(freq2angle(linspace(inputfreq,inputfreq,1),220,440),freq2angle(nB3.fchar,220,440))

nC4 = NeuronGroup(1, neuron_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = neuron_reset,method='euler')
nC4.v = c
nC4.u = c*b
nC4.fchar = 261.6 
nC4.I = neuron_tuning(freq2angle(linspace(inputfreq,inputfreq,1),220,440),freq2angle(nC4.fchar,220,440))

nD4 = NeuronGroup(1, neuron_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = neuron_reset,method='euler')
nD4.v = c
nD4.u = c*b
nD4.fchar = 293.7
nD4.I = neuron_tuning(freq2angle(linspace(inputfreq,inputfreq,1),220,440),freq2angle(nD4.fchar,220,440))

nE4 = NeuronGroup(1, neuron_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = neuron_reset,method='euler')
nE4.v = c
nE4.u = c*b
nE4.fchar = 329.6
nE4.I = neuron_tuning(freq2angle(linspace(inputfreq,inputfreq,1),220,440),freq2angle(nE4.fchar,220,440))

nF4 = NeuronGroup(1, neuron_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = neuron_reset,method='euler')
nF4.v = c
nF4.u = c*b
nF4.fchar = 349.2
nF4.I = neuron_tuning(freq2angle(linspace(inputfreq,inputfreq,1),220,440),freq2angle(nF4.fchar,220,440))

nG4 = NeuronGroup(1, neuron_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = neuron_reset,method='euler')
nG4.v = c
nG4.u = c*b
nG4.fchar = 392.0
nG4.I = neuron_tuning(freq2angle(linspace(inputfreq,inputfreq,1),220,440),freq2angle(nG4.fchar,220,440))

si=np.linspace(220,440,1000)
curveA3=neuron_tuning(freq2angle(si,220,440),freq2angle(nA3.fchar,220,440))
curveB3=neuron_tuning(freq2angle(si,220,440),freq2angle(nB3.fchar,220,440))
curveC4=neuron_tuning(freq2angle(si,220,440),freq2angle(nC4.fchar,220,440))
curveD4=neuron_tuning(freq2angle(si,220,440),freq2angle(nD4.fchar,220,440))
curveE4=neuron_tuning(freq2angle(si,220,440),freq2angle(nE4.fchar,220,440))
curveF4=neuron_tuning(freq2angle(si,220,440),freq2angle(nF4.fchar,220,440))
curveG4=neuron_tuning(freq2angle(si,220,440),freq2angle(nG4.fchar,220,440))

# tuning curves
figure(1)
clf()
hold(True)
plot(si,curveA3)
plot(si,curveB3)
plot(si,curveC4)
plot(si,curveD4)
plot(si,curveE4)
plot(si,curveF4)
plot(si,curveG4)

# spike monitor
MA3 = StateMonitor(nA3, ('v','I'), record=True)
MB3 = StateMonitor(nB3, ('v','I'), record=True)
MC4 = StateMonitor(nC4, ('v','I'), record=True)
MD4 = StateMonitor(nD4, ('v','I'), record=True)
ME4 = StateMonitor(nE4, ('v','I'), record=True)
MF4 = StateMonitor(nF4, ('v','I'), record=True)
MG4 = StateMonitor(nG4, ('v','I'), record=True)

run(10*ms,report='text')

figure(2)
clf()
subplot(4,2,1)
plot(MA3.t/ms, MA3.v[0])
subplot(4,2,2)
plot(MB3.t/ms, MB3.v[0])
subplot(4,2,3)
plot(MC4.t/ms, MC4.v[0])
subplot(4,2,4)
plot(MD4.t/ms, MD4.v[0])
subplot(4,2,5)
plot(ME4.t/ms, ME4.v[0])
subplot(4,2,6)
plot(MF4.t/ms, MF4.v[0])
subplot(4,2,7)
plot(MG4.t/ms, MG4.v[0])

figure(3)
clf()
subplot(4,2,1)
plot(MA3.t/ms, MA3.I[0])
subplot(4,2,2)
plot(MB3.t/ms, MB3.I[0])
subplot(4,2,3)
plot(MC4.t/ms, MC4.I[0])
subplot(4,2,4)
plot(MD4.t/ms, MD4.I[0])
subplot(4,2,5)
plot(ME4.t/ms, ME4.I[0])
subplot(4,2,6)
plot(MF4.t/ms, MF4.I[0])
subplot(4,2,7)
plot(MG4.t/ms, MG4.I[0])