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
import os, pickle
import skimage.measure


pool = (25, 20)
epochs = 5



defaultclock.dt=0.1*ms
taupre = taupost = 20*ms
wmax = 2*.8
Apre = 0.01
Apost = -Apre*taupre/taupost*1.05

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

def fullyConnect(gr1, gr2):
    
    syn = Synapses(gr1, gr2, clock=gr1.clock,method='euler',model='''
            	w : 1
            	dapre/dt = -apre/taupre : 1 (clock-driven) 
            	dapost/dt = -apost/taupost : 1 (clock-driven)
                heb : 1
                stdp : 1
            	''',
    		on_pre='''
    		z += w
    		apre += ((Apre)*(heb) - Apre*(not heb))*(stdp)
            w = clip(w+apost, 0, wmax)
    		''',
    		on_post='''
    		apost += ((Apost)*(heb) - Apost*(not heb))*(stdp)
    		w = clip(w+apre, 0, wmax)
    		''')

    
    iv = np.repeat(np.arange(gr1.N), gr2.N)
    jv = np.tile(np.arange(gr2.N), gr1.N)
    syn.connect(i = iv, j = jv)
    
    syn.heb = 1
    syn.stdp = 1
    syn.w = .1
    
    return syn


def convConnect(gr1, gr2, imgsize, width, height):
    
    ##UNUSED##

    syn = Synapses(gr1, gr2, clock=gr1.clock,method='euler',model='''
            	w : 1
            	dapre/dt = -apre/taupre : 1 (clock-driven) 
            	dapost/dt = -apost/taupost : 1 (clock-driven)
                heb : 1
                stdp : 1
            	''',
    		on_pre='''
    		z += w
    		apre += (Apre)#*(heb) - Apre*(not heb))*(stdp)
            w = clip(w+apost, 0, wmax)
    		''',
    		on_post='''
    		apost += (Apost)#*(heb) - Apost*(not heb))*(stdp)
    		w = clip(w+apre, 0, wmax)
    		''')
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
    
    syn.heb = 1
    syn.stdp = 1
    syn.w = .5
    
    
    return syn

def train(inNet, duration, trainingImgs, trainingClass, epoch = None):
    if(epoch is None):
        epoch = 1
    inNet.set_states({"synapses":{"stdp":np.ones(inNet.get_states()["synapses"]["N"])}})
    inNet.set_states({"synapses_pre":{"stdp":np.ones(inNet.get_states()["synapses"]["N"])}})
    inNet.set_states({"synapses_post":{"stdp":np.ones(inNet.get_states()["synapses"]["N"])}})
    for k in range(epoch):
        for l in range(len(trainingClass)):
            inNet.set_states({"neurongroup":{"Iin":trainingImgs[l]}})
            ind = inNet.get_states()["synapses"]["j"]
            ind[ind == trainingClass[l]] = -1
            ind[ind != -1] = 0
            ind[ind == -1] = 1
            inNet.set_states({"synapses":{"heb":ind}})
            inNet.set_states({"synapses_pre":{"heb":ind}})
            inNet.set_states({"synapses_post":{"heb":ind}})
            inNet.run(duration)
            inNet.set_states({"neurongroup":{"Iin":np.zeros(len(trainingImgs[l]))}})
            inNet.run(duration)

def test(inNet, duration, testImgs, testClass):
    acc = []
    inNet.set_states({"synapses":{"stdp":np.zeros(inNet.get_states()["synapses"]["N"])}})
    inNet.set_states({"synapses_pre":{"stdp":np.zeros(inNet.get_states()["synapses"]["N"])}})
    inNet.set_states({"synapses_post":{"stdp":np.zeros(inNet.get_states()["synapses"]["N"])}})
    print("Pred", "Exp", sep="\t")
    for k in range(len(testImgs)):
        Sm = SpikeMonitor(inNet["neurongroup_1"], record=True)
        inNet.add(Sm)
        inNet.set_states({"neurongroup":{"Iin":testImgs[k]}})
        inNet.run(duration)
        s = array(Sm.count)
        s = np.where(s == max(s))
        val = []
        for x in s:
            for xv in x:
                val.append(xv)
        print(str(val), str(testClass[k]), sep="\t")
        inNet.set_states({"neurongroup":{"Iin":np.zeros(len(testImgs[k]))}})
        inNet.run(duration)
        inNet.remove(Sm)
        del Sm
        
        if(len(val) == 1):
            acc.append(val == testClass[k])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return (1.0*sum(acc)/k)
    
    
######################################
#################CODE#################
######################################

x_train = []
y_train = []
x_test = []
y_test = []

# the data, shuffled and split between train and test sets
os.chdir("./output/")
dirnames = os.listdir()
for n in dirnames:
    fname = os.listdir(n)
    random.shuffle(fname)
    fname = fname[0:9]
    for f in range(len(fname)):
        fn = fname[f]
        file = open(n+"/"+fn, 'rb')
        xtmp = pickle.load(file)
        ytmp = pickle.load(file)
        dim = pickle.load(file)
        file.close()
        tmp = np.zeros(dim)
        for i in range(len(xtmp)):
            tmp[ytmp[i]][xtmp[i]] = 1
        tmp = tmp.flatten()
        if(f < len(fname)/10):
            x_test.append(tmp)
            y_test.append(dirnames.index(n))
        else:
            x_train.append(tmp)
            y_train.append(dirnames.index(n))
            
x_train = np.array(x_train).astype('int')
y_train = np.array(y_train).astype('int')
x_test = np.array(x_test).astype('int')
y_test = np.array(y_test).astype('int')

x_tmp = []
for x in range(len(x_train)):
    tmp = x_train[x].reshape(dim)
    tmp = skimage.measure.block_reduce(tmp, pool, np.average)
    x_tmp.append(tmp.flatten())
x_train = np.array(x_tmp)
    
x_tmp = []
for x in range(len(x_test)):
    tmp = x_test[x].reshape(dim)
    tmp = skimage.measure.block_reduce(tmp, pool, np.average)
    x_tmp.append(tmp.flatten())
x_test = np.array(x_tmp)


        
num_classes = len(dirnames)
num_neurons = len(x_train[0])


inLayer = createGroup(num_neurons, 5*pool[0]*pool[1])
outLayer = createGroup(num_classes, 0)
conn = fullyConnect(inLayer, outLayer)

net = Network()
net.add(inLayer)
net.add(outLayer)
net.add(conn)

accTest = []
accTrain = []

for e in range(epochs):
    print(str(e))
    accTest.append(test(net, 100*ms, x_test, y_test))
    accTrain.append(test(net, 100*ms, x_train, y_train))
    train(net, 100*ms, x_train, y_train)
accTest.append(test(net, 100*ms, x_test, y_test))
accTrain.append(test(net, 100*ms, x_train, y_train))

os.chdir('..')
if(not os.path.isdir("./networks")):
    os.mkdir("./networks")
os.chdir("./networks")
net.store(filename="SNN.net")