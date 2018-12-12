#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 21:57:40 2018

@author: chenriq
"""
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D
from keras import backend as K
import matplotlib.pyplot as plt
import skimage.measure

import pickle, random, os
import numpy as np

# neuron density
epochs = 12
pool = (25,20)



batch_size = 4

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
    ftest = fname[:int(len(fname)/10)]
    ftrain = fname[int(len(fname)/10):]
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
            y_test.append([0]*len(dirnames))
            y_test[-1][dirnames.index(n)] = 1
        else:
            x_train.append(tmp)
            y_train.append([0]*len(dirnames))
            y_train[-1][dirnames.index(n)] = 1
    
os.chdir("..")
        
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

        
x_train = np.array(x_train).astype('bool')
y_train = np.array(y_train).astype('bool')
x_test = np.array(x_test).astype('bool')
y_test = np.array(y_test).astype('bool')
        
num_classes = len(dirnames)
num_neurons = len(x_train[0])



model = Sequential()
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
classes = model.predict(x_test, batch_size=128)

if(not os.path.isdir("./networks")):
    os.mkdir("./networks")
os.chdir("./networks")
model.save('Keras.h5')