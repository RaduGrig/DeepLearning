# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 22:19:57 2017

@author: RaduGrig
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [[0], [1], [1], [0]]
x_test = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_test = [[0], [1], [1], [0]]

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(4, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('tanh'))

sgd = SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=True)
model.compile(loss='poisson',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=10,
          batch_size=4)
score = model.evaluate(x_test, y_test, batch_size=4)
print(score)
