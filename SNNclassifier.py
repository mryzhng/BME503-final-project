# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:35:35 2018

@author: Fairc
"""

#https://www.sciencedirect.com/science/article/pii/S0925231217303119

from brian2 import *
import numpy as np
import random
import matplotlib

defaultclock.dt=0.1*ms

def createGroup(n, toggleIn, inFun = None):
    if(inFun is None):
        inFun = np.repeat(0,n)
    
    tau_rise = tau_fall =2*ms
    vR = 80
    eqs = '''
    dv/dt = (0.04*v**2 + 5*v + 140 - u + inp*Iin + g*(vR - v))/ms: 1
    du/dt = a*(b*v - u)/ms : 1
    dg/dt = -g/tau_rise + z/ms : 1
    dz/dt = -z/tau_fall : 1
    a = 0.02 : 1
    b = 0.2 : 1
    c = -65 : 1
    d = 8 : 1
    inp: 1
    Iin : 1
    tau_rise : 1/Hz
    tau_fall : 1/Hz
    vR : 1
    '''
    
    gr = NeuronGroup(n,eqs, clock=Clock(defaultclock.dt), threshold = 'v >= 30', method='euler',reset = '''
	v = c
	u = u + d 
    ''')
    gr.inp = toggleIn
    gr.tau_rise = tau_rise
    gr.tau_fall = tau_fall
    gr.vR = vR
    gr.Iin = inFun
    
    gr.v = gr.c
    gr.u = gr.b*gr.v
    
    return gr

def createSynapse(gr1, gr2):
    taupre = taupost = 20*ms
    wmax = 2*.8
    Apre = 0.01
    Apost = -Apre*taupre/taupost*1.05
    heb = 1
    
    Synp = Synapses(gr1, gr2, clock=Clock(defaultclock.dt),method='euler',model='''
            	w : 1
            	dapre/dt = -apre/taupre : 1 (clock-driven) 
            	dapost/dt = -apost/taupost : 1 (clock-driven)
                taupre : 1/Hz
                taupost : 1/Hz
                wmax : 1
                Apre : 1
                Apost : 1
                heb : 1
            	''',
    		on_pre='''
    		z += w
    		apre += Apre*(heb) - Apre*(!heb)
            w = clip(w+apost, 0, wmax)
    		''',
    		on_post='''
    		apost += Apost*(heb) - Apost*(!heb)
    		w = clip(w+apre, 0, wmax)
    		''')
    return Synp

def fullyConnect(gr1, gr2):
    syn = createSynapse(gr1, gr2)
    
    iv = np.repeat(np.arange(gr1.N), gr2.N)
    j0 = np.arange(gr2.N)
    jv = j0
    for _ in range(gr1.N-1):
        jv = np.append(jv, j0) 
    syn.connect(i = iv, j = jv)
    
    return syn


def convConnect(gr1, gr2, imgsize, width, height):
    syn = createSynapse(gr1, gr2)
    
    iv = []
    jv = []
    
    spnw = 1.0*(width-1)/2
    spnh = 1.0*(height-1)/2
    
    coords = []
    x = spnw
    y = spnh
    while(y < imgsize[0] + spnh):
        coords.append([x, y])
        x += width
        if(x >= imgsize[1] + spnw):
            x = spnw
            y += height
            
    if(gr2.N < len(coords)):
        raise Exception("Not enough nodes in outgoing layer!")
    if(gr2.N > len(coords)):
        Warning("Outgoing layer has too many nodes, some may be unused!")
    
    dw = (np.array(range(width))-np.repeat(spnw, width))
    dh = (np.array(range(height))-np.repeat(spnh, height))
    
    for k in range(len(coords)):
        x,y = coords[k]
        for l in dh:
            itmp = []
            itmp = np.repeat(x,width) + dw + (l+y)*imgsize[1]
            jtmp = np.repeat(k,width)
            
            dx = np.repeat(x, width) + dw
            while len(dx) and dx[0] < 0:
                itmp = itmp[1:]
                jtmp = jtmp[1:]
                dx = dx[1:]
                
            while len(dx) and dx[-1] >= imgsize[1]:
                itmp = itmp[:-1]
                jtmp = jtmp[:-1]
                dx = dx[:-1]
                
            while len(jtmp) and jtmp[0] < 0:
                itmp = itmp[1:]
                jtmp = jtmp[1:]
                
            while len(jtmp) and jtmp[-1] >= imgsize[0]:
                itmp = itmp[:-1]
                jtmp = jtmp[:-1]
                
            iv = np.append(iv,itmp)
            jv = np.append(jv,jtmp)
        
    iv = iv.astype(int).tolist()
    jv = jv.astype(int).tolist()
    
    syn.connect(i = iv, j = jv)
    
    return syn

    
##############################################################################
##############################################################################
##############################################################################
random.seed(100)
    
imgsize = [3, 3]
Img = np.array([random.random() for _ in range(imgsize[1]*imgsize[0])])
Img2 = Img.reshape(imgsize)
figure(1)
imshow(Img2, cmap='gray')    


inLayer = createGroup(len(Img), 100, Img)
hLayer1 = createGroup(len(Img), 0)
outLayer = createGroup(9, 100)
#conn = fullyConnect(g1, g2)
#conn = lineConnect(g1, g2, imgsize, 3)
conn1 = convConnect(inLayer, hLayer1, imgsize, 3, 1)
conn2 = fullyConnect(hLayer1, outLayer)

net = Network(inLayer, hLayer1, outLayer, conn1, conn2)
##############################################################################
##############################################################################
##############################################################################

M = StateMonitor(inLayer, ('v'), record=True)
M2 = StateMonitor(hLayer1, ('v', 'g'), record=True)
S = SpikeMonitor(inLayer)
S2 = SpikeMonitor(hLayer1)
Net.run(10*ms)

figure(2)
for i in range(len(Img)):
    plot(M.t/ms, M.v[i])
legend([str(i) for i in range(len(Img))])

figure(3)
for i in range(len(Img)):
    plot(M2.t/ms, M2.v[i])
legend([str(i) for i in range(len(Img))])



###HERE BE PSEUDO/UNTESTED CODE###

#For training can do something similar to linked paper
#Use heb = 1 for linked into expected, heb = 0 otherwise
def train(inNet, duration, trainingImgs, trainingClass, epoch):
    for k in range(epoch):
        for l in range(len(trainingClass))
            inNet.inLayer.Iin = trainingImgs[l]
            inV = np.repmat(0,len(trainingClass))
            inV[l] = 1
            inNet.outLayer.Iin = inV
            inNet.run(duration)
            inNet.outLayer.Iin = np.repmat(0,len(trainingClass))
            inNet.run(duration)

def test(inNet, duration, testImgs, testClass):
    #Can check compared to testclass or overall
    #Might need way to toggle off STDP during testing?
    Sm = SpikeMonitor(inNet.outLayer)
    for k in range(testImgs):
        inNet.inLayer.Iin = testImgs[k]
        inNet.run(duration)
        val = SpikeMonitor.num_spikes.index(max(SpikeMonitor.num_spikes))
        
