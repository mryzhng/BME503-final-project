#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 21:16:15 2018

@author: talafakhoury
"""

## Tuning curve front end code
# Neurons should be only sensitive to a specific range -a to a around the preferred frequency
# otherwise activation should be equal to 0

# code currently modulates input current to 
from brian2 import *
import matplotlib.pyplot as plt

# input freq
inputfreq = array([240, 240, 400])

# tuning curve function
def neuron_tuning(s,fpref):
    g=30 # maximum firing rate at prefered frequency
    at= 0.8
    lengthsi=size(s) # s is the x vector or different frequency values in a range going from -p 
    current=0.0*linspace(-pi,pi,lengthsi) # initialize dummy array to replace with firing value = used as current input
    for i in range(0,lengthsi):
       if cos(s[i]-fpref) > at:
           current[i]=g*(cos(s[i]-fpref)-at) # cos(s-spref) should yield cos(0)=1 max value when s = spref
       else:
           current[i]=0
    return current


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

nB3 = NeuronGroup(1, neuron_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = neuron_reset,method='euler')
nB3.v = c
nB3.u = c*b
nB3.fchar = 246.9

nC4 = NeuronGroup(1, neuron_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = neuron_reset,method='euler')
nC4.v = c
nC4.u = c*b
nC4.fchar = 261.6 

nD4 = NeuronGroup(1, neuron_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = neuron_reset,method='euler')
nD4.v = c
nD4.u = c*b
nD4.fchar = 293.7

nE4 = NeuronGroup(1, neuron_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = neuron_reset,method='euler')
nE4.v = c
nE4.u = c*b
nE4.fchar = 329.6

nF4 = NeuronGroup(1, neuron_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = neuron_reset,method='euler')
nF4.v = c
nF4.u = c*b
nF4.fchar = 349.2

nG4 = NeuronGroup(1, neuron_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = neuron_reset,method='euler')
nG4.v = c
nG4.u = c*b
nG4.fchar = 392.0


for i in range (0, inputfreq.size):
    nA3.I = nA3.I + neuron_tuning((linspace(inputfreq[i],inputfreq[i],1),220,440),(nA3.fchar,220,440))
    nB3.I = nB3.I + neuron_tuning((linspace(inputfreq[i],inputfreq[i],1),220,440),(nB3.fchar,220,440))
    nC4.I = nC4.I + neuron_tuning((linspace(inputfreq[i],inputfreq[i],1),220,440),(nC4.fchar,220,440))
    nD4.I = nD4.I + neuron_tuning((linspace(inputfreq[i],inputfreq[i],1),220,440),(nD4.fchar,220,440))
    nE4.I = nE4.I + neuron_tuning((linspace(inputfreq[i],inputfreq[i],1),220,440),(nE4.fchar,220,440))
    nF4.I = nF4.I + neuron_tuning((linspace(inputfreq[i],inputfreq[i],1),220,440),(nF4.fchar,220,440))
    nG4.I = nG4.I + neuron_tuning((linspace(inputfreq[i],inputfreq[i],1),220,440),(nG4.fchar,220,440))

si=np.linspace(200,500,1000) # range of frequencies we are testing the neurons to
curveA3=neuron_tuning((si,220,440),(nA3.fchar,220,440))
curveB3=neuron_tuning((si,220,440),(nB3.fchar,220,440))
curveC4=neuron_tuning((si,220,440),(nC4.fchar,220,440))
curveD4=neuron_tuning((si,220,440),(nD4.fchar,220,440))
curveE4=neuron_tuning((si,220,440),(nE4.fchar,220,440))
curveF4=neuron_tuning((si,220,440),(nF4.fchar,220,440))
curveG4=neuron_tuning((si,220,440),(nG4.fchar,220,440))

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

smA3 = SpikeMonitor(nA3)
smB3 = SpikeMonitor(nB3)
smC4 = SpikeMonitor(nC4)
smD4 = SpikeMonitor(nD4)
smE4 = SpikeMonitor(nE4)
smF4 = SpikeMonitor(nF4)
smG4 = SpikeMonitor(nG4)

run(100*ms,report='text')

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

figure(4)
clf()
hold(True)
plot(smA3.t/ms, smA3.i+1, 'k*')
plot(smB3.t/ms, smB3.i+2, 'k*')
plot(smC4.t/ms, smC4.i+3, 'k*')
plot(smD4.t/ms, smD4.i+4, 'k*')
plot(smE4.t/ms, smE4.i+5, 'k*')
plot(smF4.t/ms, smF4.i+6, 'k*')
plot(smG4.t/ms, smG4.i+7, 'k*')
ylim(0, 8)
yticks(np.arange(8), ('', 'A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4'))


